/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by KNIME AG, Zurich, Switzerland
 *  Website: http://www.knime.com; Email: contact@knime.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 * History
 *   Oct 1, 2018 (simon): created
 */
package org.knime.base.node.meta.feature.selection.genetic;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.Predicate;

import org.knime.base.node.meta.feature.selection.FeatureSelectionStrategy;
import org.knime.core.node.InvalidSettingsException;

import io.jenetics.BitChromosome;
import io.jenetics.BitGene;
import io.jenetics.EliteSelector;
import io.jenetics.Genotype;
import io.jenetics.Mutator;
import io.jenetics.Optimize;
import io.jenetics.Selector;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.EvolutionStart;
import io.jenetics.util.Factory;
import io.jenetics.util.RandomRegistry;

/**
 * Genetic Algorithm for Feature Selection. Uses the Jenetics library and runs a separate thread for the genetic
 * algorithm.
 *
 * @author Simon Schmid, KNIME, Austin, USA
 */
public class GeneticStrategy implements FeatureSelectionStrategy {

    private final int m_maxNumIterations;

    private boolean m_isMinimize;

    private boolean m_continueLoop = true;

    private boolean m_continueWorkflowLoop;

    private final Evaluator m_evaluator = new Evaluator();

    private final Lock m_lock = new ReentrantLock();

    private final Condition m_condScoreReceived = m_lock.newCondition();

    private final Condition m_condContinueLoop = m_lock.newCondition();

    private final Condition m_condGAInitialized = m_lock.newCondition();

    private final ExecutorService m_executor;

    private RuntimeException m_exception;

    private Engine<BitGene, Double> m_engine;

    /**
     * Constructor. Starts the thread for the genetic algorithm already to be able to give an output of the first
     * selected features during configure.
     *
     * @param subSetSize max number of selected features, is <= 0 if undefined
     * @param popSize population size
     * @param maxNumGenerations max number of generations
     * @param useSeed if seed should be used
     * @param seed the seed
     * @param crossoverRate the crossover rate
     * @param mutationRate the mutation rate
     * @param elitismRate the elitism rate
     * @param selectionStrategy the selection strategy
     * @param crossoverStrategy the crossover strategy
     * @param features ids of the features
     *
     */
    public GeneticStrategy(final int subSetSize, final int popSize, final int maxNumGenerations, final boolean useSeed,
        final long seed, final double crossoverRate, final double mutationRate, final double elitismRate,
        final SelectionStrategy selectionStrategy, final CrossoverStrategy crossoverStrategy,
        final List<Integer> features) {
        if (features.size() < 2) {
            throw new IllegalArgumentException(
                "To use a genetic algorithm, the number of features must be at least 2.");
        }
        // probably it's going to be less, this is just an upper bound
        m_maxNumIterations = maxNumGenerations * popSize;

        final Random random;
        if (useSeed) {
            random = new Random(seed);
        } else {
            random = new Random();
        }

        /*
         * This thread will run the genetic algorithm (ga) which will then run in parallel with the workflow thread. Those two threads will have to wait for each other. The workflow thread needs to wait for the ga thread to produce a feature subset to run and the ga thread needs to wait for the workflow loop to score this subset.
         */
        final Runnable runnable = new Runnable() {

            @Override
            public void run() {
                try {
                    // 1.) Define the genotype (factory) suitable for the problem.
                    final double probOfTrues;
                    if (subSetSize <= 0) {
                        probOfTrues = 0.5;
                    } else {
                        // set it to get subsets of approximately the half size of the max size
                        probOfTrues = ((double)Math.min(subSetSize, features.size()) / 2) / features.size();
                    }
                    final Factory<Genotype<BitGene>> gtf = Genotype.of(BitChromosome.of(features.size(), probOfTrues));

                    // 2.) Define a validator for the genotype which ensures the maximal number of selected features.
                    final Predicate<? super Genotype<BitGene>> validator = gt -> {
                        final int bitCount = gt.get(0).as(BitChromosome.class).bitCount();
                        if (subSetSize <= 0) {
                            return bitCount > 1;
                        }
                        return bitCount > 1 && bitCount <= subSetSize;
                    };

                    // 3.) Create the execution environment.
                    m_engine = RandomRegistry.with(new Random(random.nextLong()), f -> {
                        return Engine.builder(m_evaluator, gtf).executor(Runnable::run).populationSize(popSize)
                            .alterers(CrossoverStrategy.getSelector(crossoverStrategy, crossoverRate),
                                new Mutator<>(mutationRate))
                            // most likely we don't know yet whether to minimize oder maximize, must be changed
                            // later using reflection (see #setIsMinimize)
                            .optimize(m_isMinimize ? Optimize.MINIMUM : Optimize.MAXIMUM)
                            .selector(getSelector(selectionStrategy, (int)(elitismRate * popSize + 0.5)))
                            // after 100 retries, the max subset size will be ignored
                            .genotypeValidator(validator).individualCreationRetries(100).build();
                    });

                    // 4.) Start the execution (evolution) and collect the result.
                    EvolutionResult<BitGene, Double> result =
                        RandomRegistry.with(new Random(random.nextLong()), f -> m_engine.stream()
                            .limit(maxNumGenerations).collect(EvolutionResult.toBestEvolutionResult()));
                    RandomRegistry.with(new Random(random.nextLong()),
                        f -> m_engine.stream(EvolutionStart.of(result.getPopulation(), result.getGeneration()))
                            .limit(maxNumGenerations - 1).collect(EvolutionResult.toBestEvolutionResult()));

                    m_continueLoop = false;
                    m_continueWorkflowLoop = true;
                    m_lock.lock();
                    m_condContinueLoop.signal();
                    m_lock.unlock();
                } catch (RuntimeException e) {
                    // signal all potentially awaiting conditions, set an exception to throw
                    m_executor.shutdown();
                    m_exception = e;
                    m_lock.lock();
                    m_condContinueLoop.signal();
                    m_condGAInitialized.signal();
                    m_lock.unlock();
                }
            }
        };

        m_executor = Executors.newSingleThreadExecutor(new ThreadFactory() {

            @Override
            public Thread newThread(final Runnable r) {
                return new Thread(r, "Genetic Algorithm");
            }
        });
        m_executor.execute(runnable);
    }

    private static Selector<BitGene, Double> getSelector(final SelectionStrategy selectionStrategy,
        final int eliteCount) {
        final Selector<BitGene, Double> nonElitistSelector = SelectionStrategy.getSelector(selectionStrategy);
        final Selector<BitGene, Double> selector;
        if (eliteCount > 0) {
            selector = new EliteSelector<BitGene, Double>(eliteCount, nonElitistSelector);
        } else {
            selector = nonElitistSelector;
        }
        return selector;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean continueLoop() {
        // wait until the genetic algorithm thread has selected the next feature subset
        m_lock.lock();
        try {
            while (!m_continueWorkflowLoop) {
                // if the genetic algorithm thread has been interrupted in some way, throw the exception
                if (m_exception != null) {
                    throw m_exception;
                }
                m_condContinueLoop.await();
            }
        } catch (InterruptedException e) {
            // nothing to do
        } finally {
            m_lock.unlock();
        }
        m_continueWorkflowLoop = false;

        if (!m_continueLoop) {
            m_executor.shutdown();
        }
        return m_continueLoop;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Integer> getIncludedFeatures() {
        return m_evaluator.getCurrent();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addScore(final double score) {
        m_evaluator.setScore(score);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setIsMinimize(final boolean isMinimize) {
        try {
            // The only way to change whether to minimize or maximize after the genetic algorithm has been started
            // is reflection. Since we know how to optimize not before the Loop End has been configured and the genetic
            // algorithm needs to be started during configure of the Loop Start, the optimization method may change.
            final Field f = m_engine.getClass().getDeclaredField("_optimize");
            f.setAccessible(true);
            // Remove final modifier
            final Field modifiersField = Field.class.getDeclaredField("modifiers");
            modifiersField.setAccessible(true);
            modifiersField.setInt(f, f.getModifiers() & ~Modifier.FINAL);
            // Set optimization method
            f.set(m_engine, isMinimize ? Optimize.MINIMUM : Optimize.MAXIMUM);
            m_isMinimize = isMinimize;
        } catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean shouldAddFeatureLevel() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getCurrentlyBestScore() {
        return m_evaluator.getScore();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void prepareNewRound() {
        // nothing to prepare
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Integer> getFeatureLevel() {
        return m_evaluator.getCurrent();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getNameForLastChange() {
        return "Selected features";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Integer> getLastChange() {
        m_lock.lock();
        final List<Integer> current = m_evaluator.getCurrent();
        // this function is the last called function before a new round is started, therefore continue the genetic algorithm
        m_condScoreReceived.signal();
        m_lock.unlock();
        return current;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getNumberOfIterations() {
        return m_maxNumIterations;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Integer getCurrentFeature() {
        // just needed for flow variables, this strategy does not have a current feature
        return -1;
    }

    private final class Evaluator implements Function<Genotype<BitGene>, Double> {

        // caches the scores of already scored feature subsets
        private final HashMap<Integer, Double> m_scoreLookUp = new HashMap<>();

        private boolean m_scoreReceived;

        private double m_score;

        private List<Integer> m_current;

        /**
         * {@inheritDoc}
         */
        @Override
        public synchronized Double apply(final Genotype<BitGene> genotype) {
            final int hashCode = genotype.get(0).as(BitChromosome.class).toBitSet().hashCode();
            // if the genotype has already been processed earlier, return the cached score
            if (m_scoreLookUp.containsKey(hashCode)) {
                return m_scoreLookUp.get(hashCode);
            }

            // a new feature subset needs to be scored, tell the Loop End node to continue the loop
            m_lock.lock();
            if (m_current != null) {
                setCurrent(genotype);
            } else {
                setInitial(genotype);
            }
            m_continueWorkflowLoop = true;
            m_condContinueLoop.signal();
            m_lock.unlock();
            m_scoreReceived = false;

            // wait until the loop finishes and the score is received
            m_lock.lock();
            try {
                while (!m_scoreReceived) {
                    m_condScoreReceived.await();
                }
            } catch (InterruptedException e) {
                // nothing to do
            } finally {
                m_lock.unlock();
            }

            // cache the received score to speed up further computation
            m_scoreLookUp.put(hashCode, m_score);
            return m_score;
        }

        private synchronized void setCurrent(final Genotype<BitGene> genotype) {
            m_current.clear();
            for (int i = 0; i < genotype.getChromosome(0).length(); i++) {
                if (genotype.getChromosome(0).getGene(i).booleanValue()) {
                    m_current.add(i);
                }
            }
        }

        private synchronized void setInitial(final Genotype<BitGene> genotype) {
            m_lock.lock();
            if (m_current == null) {
                m_current = new ArrayList<>();
            }
            setCurrent(genotype);
            m_condGAInitialized.signal();
            m_lock.unlock();
        }

        /**
         * @return the currentGenotype
         */
        private List<Integer> getCurrent() {
            if (m_current != null) {
                return new ArrayList<>(m_current);
            }

            // wait before the initial genotype is set before the Loop Start can ask for the first feature subset
            m_lock.lock();
            while (m_current == null) {
                try {
                    if (m_exception != null) {
                        throw m_exception;
                    }
                    m_condGAInitialized.await();
                } catch (InterruptedException e) {
                    // nothing to do
                }
            }
            m_lock.unlock();
            return new ArrayList<>(m_current);
        }

        /**
         * @return the score
         */
        public double getScore() {
            return m_score;
        }

        /**
         * @param score the score to set
         */
        private void setScore(final double score) {
            m_score = score;
            m_scoreReceived = true;
        }

    }

}

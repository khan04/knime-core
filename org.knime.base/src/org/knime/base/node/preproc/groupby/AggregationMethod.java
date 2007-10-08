/*
 * -------------------------------------------------------------------
 * This source code, its documentation and all appendant files
 * are protected by copyright law. All rights reserved.
 *
 * Copyright, 2003 - 2007
 * University of Konstanz, Germany
 * Chair for Bioinformatics and Information Mining (Prof. M. Berthold)
 * and KNIME GmbH, Konstanz, Germany
 *
 * You may not modify, publish, transmit, transfer or sell, reproduce,
 * create derivative works from, distribute, perform, display, or in
 * any way exploit any of the content, in whole or in part, except as
 * otherwise expressly permitted in writing by the copyright owner or
 * as specified in the license file distributed with this product.
 *
 * If you have any questions please contact the copyright holder:
 * website: www.knime.org
 * email: contact@knime.org
 * -------------------------------------------------------------------
 * 
 * History
 *    28.06.2007 (Tobias Koetter): created
 */

package org.knime.base.node.groupby;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataType;
import org.knime.core.data.DoubleValue;
import org.knime.core.data.def.DoubleCell;
import org.knime.core.data.def.IntCell;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.util.MutableInteger;


/**
 * Enumeration which list all available aggregation methods including helper
 * methods.
 * 
 * @author Tobias Koetter, University of Konstanz
 */
public enum AggregationMethod {
    /**Minimum.*/
    MIN("Minimum", true, "Min({1})", DoubleCell.TYPE, false),
    /**Maximum.*/
    MAX("Maximum", true, "Max({1})", DoubleCell.TYPE, false),
    /**Average.*/
    MEAN("Mean", true, "Mean({1})", DoubleCell.TYPE, false),
    /**Sum.*/
    SUM("Sum", true, "Sum({1})", DoubleCell.TYPE, false),
    /**Variance.*/
    VARIANCE("Variance", true, "Variance({1})", DoubleCell.TYPE, false),
    
//The none numerical methods    
    /**Takes the first cell per group.*/
    FIRST("First", false, "First({1})", null, false),
    /**Takes the last cell per group.*/
    LAST("Last", false, "Last({1})", null, false),
    /**Takes the value which occurs most.*/
    MODE("Mode", false, "Mode({1})", null, true),
    /**Counts the number of group members.*/
    COUNT("Count", false, "Count({1})", IntCell.TYPE, false);
    
    private static final String PLACE_HOLDER = "{1}";
    
    private final class MinOperator extends AggregationOperator {
        
        private double m_minVal = Double.NaN;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public MinOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            final double d = ((DoubleValue)cell).getDoubleValue();
            if (Double.isNaN(m_minVal) || d < m_minVal) {
                m_minVal = d;
            }
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (Double.isNaN(m_minVal)) {
                return DataType.getMissingCell();
            }
            return new DoubleCell(m_minVal);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_minVal = Double.NaN;
        }   
    }
    
    private final class MaxOperator extends AggregationOperator {
        
        private double m_maxVal = Double.NaN;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public MaxOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            final double d = ((DoubleValue)cell).getDoubleValue();
            if (Double.isNaN(m_maxVal) || d > m_maxVal) {
                m_maxVal = d;
            }
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (Double.isNaN(m_maxVal)) {
                return DataType.getMissingCell();
            }
            return new DoubleCell(m_maxVal);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_maxVal = Double.NaN;
        }
    }
    
    private final class MeanOperator extends AggregationOperator {

        private double m_sum = 0;
        private int m_count = 0;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public MeanOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            final double d = ((DoubleValue)cell).getDoubleValue();
            m_sum += d;
            m_count++;
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (m_count == 0) {
                return DataType.getMissingCell();
            }
            return new DoubleCell(m_sum / m_count);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_sum = 0;
            m_count = 0;
        }   
    }
    
    private final class SumOperator extends AggregationOperator {
        private boolean m_valid = false;
        private double m_sum = 0;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public SumOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            m_valid = true;
            final double d = ((DoubleValue)cell).getDoubleValue();
            m_sum += d;
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (!m_valid) {
                return DataType.getMissingCell();
            }
            return new DoubleCell(m_sum);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_valid = false;
            m_sum = 0;
        }
    }
    
    private final class VarianceOperator extends AggregationOperator {

        /**Constructor for class VarianceOperator.
         * @param maxUniqueValues
         */
        public VarianceOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }

        private double m_sumSquare = 0;
        private double m_sum = 0;
        private int m_validCount = 0;
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            final double d = ((DoubleValue)cell).getDoubleValue();
            m_validCount++;
            m_sum += d;
            m_sumSquare += d * d;
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (m_validCount <= 1) {
                return DataType.getMissingCell();
            }
            double variance = (m_sumSquare - ((m_sum * m_sum) 
                    / m_validCount)) / (m_validCount - 1);
            // unreported bug fix: in cases in which a column contains 
            // almost only one value (for instance 1.0) but one single 
            // 'outlier' whose value is, for instance 0.9999998, we get 
            // round-off errors resulting in negative variance values
            if (variance < 0.0 && variance > -1.0E8) {
                variance = 0.0;
            }
            return new DoubleCell(variance);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_sumSquare = 0;
            m_sum = 0;
            m_validCount = 0;
        }
        
    }
        
    private final class FirstOperator extends AggregationOperator {
        
        private DataCell m_firstCell = null;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public FirstOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            if (m_firstCell == null) {
                m_firstCell = cell;
            }
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (m_firstCell == null) {
                return DataType.getMissingCell();
            }
            return m_firstCell;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_firstCell = null;
        }
    }

    private final class LastOperator extends AggregationOperator {
        
        private DataCell m_lastCell = null;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public LastOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            m_lastCell = cell;
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (m_lastCell == null) {
                return DataType.getMissingCell();
            }
            return m_lastCell;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_lastCell = null;
        }
    }

    private final class ModeOperator extends AggregationOperator {
        
        private final Map<DataCell, MutableInteger> m_valCounter;

        /**Constructor for class MinOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public ModeOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
            try {
                m_valCounter = 
                    new HashMap<DataCell, MutableInteger>(maxUniqueValues);
            } catch (OutOfMemoryError e) {
                throw new IllegalArgumentException(
                        "Maximum unique values number to big");
            }
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            MutableInteger counter = m_valCounter.get(cell);
            if (counter == null) {
                //check if the maps contains more values than allowed
                //before adding a new value
                if (m_valCounter.size() >= getMaxUniqueValues()) {
                    return true;
                }
                counter = new MutableInteger(0);
                m_valCounter.put(cell, counter);
            }
            counter.inc();
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            if (m_valCounter.size() < 1) {
                return DataType.getMissingCell();
            }
            //get the cell with the most counts
            final Set<Entry<DataCell, MutableInteger>> entries = 
                m_valCounter.entrySet();
            int max = Integer.MIN_VALUE;
            DataCell result = null;
            for (Entry<DataCell, MutableInteger> entry : entries) {
                if (result == null || entry.getValue().intValue() > max) {
                    max = entry.getValue().intValue();
                    result = entry.getKey();
                }
            }
            return result;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            if (m_valCounter != null) {
                m_valCounter.clear();
            }
        }
    }
    
    private final class CountOperator extends AggregationOperator {
        
        private int m_counter = 0;

        /**Constructor for class CountOperator.
         * @param maxUniqueValues the maximum number of unique values
         */
        public CountOperator(final int maxUniqueValues) {
            super(maxUniqueValues);
        }
        
        /**
         * {@inheritDoc}
         */
        @Override
        protected boolean computeInternal(final DataCell cell) {
            m_counter++;
            return false;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected DataCell getResultInternal() {
            return new IntCell(m_counter);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void resetInternal() {
            m_counter = 0;
        }   
    }
    
    private final String m_label;
    private final boolean m_numerical;
    private final String m_columnNamePattern;
    private final DataType m_dataType;
    private final boolean m_usesLimit;
    
    /**Constructor for class AggregationMethod.
     * @param label user readable label
     * @param numerical <code>true</code> if the operator is only suitable
     * for numerical columns
     * @param columnNamePattern the pattern for the result column name
     * @param type the {@link DataType} of the result column
     * @param usesLimit <code>true</code> if the method checks the number of
     * unique values limit.
     */
    private AggregationMethod(final String label, final boolean numerical,
            final String columnNamePattern, final DataType type,
            final boolean usesLimit) {
        m_label = label;
        m_numerical = numerical;
        m_columnNamePattern = columnNamePattern;
        m_dataType = type;
        m_usesLimit = usesLimit;
    }

    
    /**
     * @return the label
     */
    public String getLabel() {
        return m_label;
    }

    
    /**
     * @return the numerical
     */
    public boolean isNumerical() {
        return m_numerical;
    }
    
    /**
     * @param maxUniqueValues the maximum number of unique values
     * @return the operator of this method
     */
    public AggregationOperator getOperator(final int maxUniqueValues) {
        switch (this) {
            case MIN:       return new MinOperator(maxUniqueValues);
            case MAX:       return new MaxOperator(maxUniqueValues);
            case MEAN:       return new MeanOperator(maxUniqueValues);
            case SUM:       return new SumOperator(maxUniqueValues);
            case VARIANCE:  return new VarianceOperator(maxUniqueValues);
            case FIRST:     return new FirstOperator(maxUniqueValues);
            case LAST:      return new LastOperator(maxUniqueValues);
            case MODE:      return new ModeOperator(maxUniqueValues);
            case COUNT:     return new CountOperator(maxUniqueValues);
        }
        throw new IllegalStateException("No operator found");
    }
    
    /**
     * @param origColumnName the original column name
     * @return the new name of the aggregation column
     */
    public String getColumnName(final String origColumnName) {
        if (m_columnNamePattern == null || m_columnNamePattern.length() < 1) {
            return origColumnName;
        }
        return m_columnNamePattern.replace(PLACE_HOLDER, origColumnName);
    }
    
    /**
     * @param origDataType the original {@link DataType}
     * @return the {@link DataType} of the aggregation column
     */
    public DataType getColumnType(final DataType origDataType) {
        if (m_dataType == null) {
            return origDataType;
        }
        return m_dataType;
    }
    
    
    /**
     * @return <code>true</code> if this method checks the maximum unique
     * values limit.
     */
    public boolean isUsesLimit() {
        return m_usesLimit;
    }
    
    /**
     * @return the default method for numerical columns
     */
    public static AggregationMethod getDefaultNumericMethod() {
        return MIN;
    }
    
    /**
     * @return the default method for none numerical columns
     */
    public static AggregationMethod getDefaultNoneNumericMethod() {
        return FIRST;
    }
    
    /**
     * @param model the {@link SettingsModelString} with the label of the
     * <code>AggregationMethod</code>
     * @return the <code>AggregationMethod</code>
     */
    public static AggregationMethod getMethod4SettingsModel(
            final SettingsModelString model) {
        if (model == null) {
            throw new NullPointerException("model must not be null");
        }
        return getMethod4Label(model.getStringValue());
    }
    
    /**
     * @param label the label to get the <code>AggregationMethod</code> for.
     * @return the <code>AggregationMethod</code> with the given label
     * @throws IllegalArgumentException if no <code>AggregationMethod</code>
     * exists for the given label
     */
    public static AggregationMethod getMethod4Label(final String label) 
    throws IllegalArgumentException {
        if (label == null) {
            throw new NullPointerException("Label must not be null");
        }
        final AggregationMethod[] methods = values();
        for (AggregationMethod method : methods) {
            if (method.getLabel().equals(label)) {
                return method;
            }
        }
        throw new IllegalArgumentException("No method found for label: " 
                + label);
    }
    
    /**
     * @return a <code>List</code> with the labels of all numerical methods
     */
    public static List<String> getNumericalMethodLabels() {
        final AggregationMethod[] methods = values();
        final List<String> labels = new ArrayList<String>(methods.length);
        for (int i = 0, length = methods.length; i < length; i++) {
            labels.add(methods[i].getLabel());
        }
        return labels;
    }
    
    /**
     * @return a <code>List</code> with the labels of all none numerical methods
     */
    public static List<String> getNoneNumericalMethodLabels() {
        return getLabels(false);
    }
    
    private static List<String> getLabels(final boolean numeric) {
        final AggregationMethod[] methods = values();
        final List<String> labels = new ArrayList<String>(methods.length);
        for (int i = 0, length = methods.length; i < length; i++) {
            if (methods[i].isNumerical() == numeric) {
                labels.add(methods[i].getLabel());
            }
        }
        return labels;
    }
}

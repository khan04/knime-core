/* 
 * ------------------------------------------------------------------
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
 * --------------------------------------------------------------------- *
 * 
 * History
 *   Sep 18, 2007 (wiswedel): created
 */
package org.knime.core.node.workflow;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;

import org.knime.core.data.container.ContainerTable;
import org.knime.core.eclipseUtil.GlobalClassCreator;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.GenericNodeFactory;
import org.knime.core.node.GenericNodeModel;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.Node;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodePersistor;
import org.knime.core.node.NodePersistorVersion_1xx;
import org.knime.core.node.NodeSettings;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.workflow.NodeContainer.State;
import org.knime.core.node.workflow.WorkflowPersistor.LoadResult;

/**
 * 
 * @author wiswedel, University of Konstanz
 */
class SingleNodeContainerPersistorVersion1xx implements SingleNodeContainerPersistor {

    private static final NodeLogger LOGGER = NodeLogger
            .getLogger(SingleNodeContainerPersistorVersion1xx.class);

    private Node m_node;
    
    private NodeContainerMetaPersistorVersion1xx m_metaPersistor;
    
    private final HashMap<Integer, ContainerTable> m_globalTableRepository;
    
    private NodeSettingsRO m_nodeSettings;
    private File m_nodeDir;
    
    SingleNodeContainerPersistorVersion1xx(
            final HashMap<Integer, ContainerTable> tableRep) {
        m_globalTableRepository = tableRep;
    }

    public NodeContainerMetaPersistor getMetaPersistor() {
        return m_metaPersistor;
    }

    /** {@inheritDoc} */
    public Node getNode() {
        return m_node;
    }
    
    /** {@inheritDoc} */
    public ScopeObjectStack getScopeObjectStack() {
        return null;
    }
    
    /** {@inheritDoc} */
    public SingleNodeContainer getNodeContainer(
            final WorkflowManager wm, final NodeID id) {
        return new SingleNodeContainer(wm, id, this);
    }
    
    protected NodePersistorVersion_1xx createNodePersistor() {
        return new NodePersistorVersion_1xx();
    }

    /** {@inheritDoc} */
    public LoadResult preLoadNodeContainer(final File nodeSettingsFile, 
            final ExecutionMonitor exec, final NodeSettingsRO parentSettings) 
    throws InvalidSettingsException, CanceledExecutionException, IOException {
        LoadResult result = new LoadResult();
        String error;
        if (nodeSettingsFile == null || !nodeSettingsFile.isFile()) {
            throw new IOException("Can't read node file \"" 
                    + nodeSettingsFile.getAbsolutePath() + "\"");
        }
        NodeSettingsRO settings = NodeSettings.loadFromXML(
                new BufferedInputStream(new FileInputStream(nodeSettingsFile)));
        String nodeFactoryClassName;
        try {
            nodeFactoryClassName = loadNodeFactoryClassName(
                    parentSettings, settings);
        } catch (InvalidSettingsException e) {
            if (nodeSettingsFile.getName().equals(
                    WorkflowPersistor.WORKFLOW_FILE)) {
                error = "Can't load meta flows in this version";
            } else {
                error = "Can't load node factory class name";
            }
            throw new InvalidSettingsException(error);
        }
        GenericNodeFactory<GenericNodeModel> nodeFactory;
        
        try {
            nodeFactory = loadNodeFactory(nodeFactoryClassName);
        } catch (Exception e) {
            error =  "Unable to load factory class \"" 
                + nodeFactoryClassName + "\"";
            throw new InvalidSettingsException(error);
        }
        m_node = new Node(nodeFactory);
        m_metaPersistor = createNodeContainerMetaPersistor();
        LoadResult metaResult = m_metaPersistor.load(settings);
        if (metaResult.hasErrors()) {
            result.addError(metaResult);
        }
        m_nodeSettings = settings;
        m_nodeDir = nodeSettingsFile.getParentFile();
        return result;
    }
    
    /** {@inheritDoc} */
    public LoadResult loadNodeContainer(final int loadID, 
            final ExecutionMonitor exec) throws InvalidSettingsException, 
            CanceledExecutionException, IOException {
        LoadResult result = new LoadResult();
        String nodeFileName;
        try {
            nodeFileName = loadNodeFile(m_nodeSettings);
        } catch (InvalidSettingsException e) {
            String error = 
                "Unable to load node settings file for node with ID suffix "
                    + m_metaPersistor.getNodeIDSuffix() + " (node \""
                    + m_node.getName() + "\"): " + e.getMessage();
            result.addError(error);
            LOGGER.debug(error, e);
            return result;
        }
        File nodeFile = new File(m_nodeDir, nodeFileName);
        NodePersistor nodePersistor = createNodePersistor();
        LoadResult nodeLoadResult = nodePersistor.load(
                m_node, nodeFile, exec, loadID, m_globalTableRepository);
        if (nodeLoadResult.hasErrors()) {
            result.addError(nodeLoadResult);
        }
        State nodeState;
        if (nodePersistor.isExecuted()) {
            nodeState = State.EXECUTED;
        } else if (nodePersistor.isConfigured()) {
            nodeState = State.CONFIGURED;
        } else {
            nodeState = State.IDLE;
        }
        m_metaPersistor.setState(nodeState);
        return result;
    }
    
    protected NodeContainerMetaPersistorVersion1xx 
        createNodeContainerMetaPersistor() {
        return new NodeContainerMetaPersistorVersion1xx();
    }
    
    protected String loadNodeFactoryClassName(
            final NodeSettingsRO parentSettings, final NodeSettingsRO settings)
            throws InvalidSettingsException {
        return parentSettings.getString(KEY_FACTORY_NAME);
    }

    @SuppressWarnings("unchecked")
    protected GenericNodeFactory<GenericNodeModel> loadNodeFactory(
            final String factoryClassName) throws InvalidSettingsException,
            InstantiationException, IllegalAccessException,
            ClassNotFoundException {
        // use global Class Creator utility for Eclipse "compatibility"
        try {
            GenericNodeFactory<GenericNodeModel> f = (GenericNodeFactory<GenericNodeModel>)((GlobalClassCreator
                    .createClass(factoryClassName)).newInstance());
            return f;
        } catch (ClassNotFoundException ex) {
            String[] x = factoryClassName.split("\\.");
            String simpleClassName = x[x.length - 1];

            for (String s : GenericNodeFactory.getLoadedNodeFactories()) {
                if (s.endsWith("." + simpleClassName)) {
                    GenericNodeFactory<GenericNodeModel> f = (GenericNodeFactory<GenericNodeModel>)((GlobalClassCreator
                            .createClass(s)).newInstance());
                    LOGGER.warn("Substituted '" + f.getClass().getName()
                            + "' for unknown factory '" + factoryClassName
                            + "'");
                    return f;
                }
            }
            throw ex;
        }
    }
    
    protected String loadNodeFile(final NodeSettingsRO settings)
        throws InvalidSettingsException {
        return SETTINGS_FILE_NAME;
    }

}

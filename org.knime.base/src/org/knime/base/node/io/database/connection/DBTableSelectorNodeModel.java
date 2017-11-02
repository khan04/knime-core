/*
 * ------------------------------------------------------------------------
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
 *  KNIME and ECLIPSE being a combined program, KNIME GMBH herewith grants
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
 *   25.04.2014 (thor): created
 */
package org.knime.base.node.io.database.connection;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.knime.base.util.flowvariable.FlowVariableProvider;
import org.knime.base.util.flowvariable.FlowVariableResolver;
import org.knime.core.data.DataTableSpec;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortType;
import org.knime.core.node.port.database.DatabaseConnectionPortObject;
import org.knime.core.node.port.database.DatabaseConnectionPortObjectSpec;
import org.knime.core.node.port.database.DatabaseConnectionSettings;
import org.knime.core.node.port.database.DatabasePortObject;
import org.knime.core.node.port.database.DatabasePortObjectSpec;
import org.knime.core.node.port.database.DatabaseQueryConnectionSettings;
import org.knime.core.node.port.database.reader.DBReader;

/**
 *
 * @author Thorsten Meinl, KNIME AG, Zurich, Switzerland
 */
class DBTableSelectorNodeModel extends NodeModel implements FlowVariableProvider {
    private DatabaseQueryConnectionSettings m_settings = new DatabaseQueryConnectionSettings();

    /**
     * Creates a new Database Table Selector node model.
     */
    DBTableSelectorNodeModel() {
        super(new PortType[] {DatabaseConnectionPortObject.TYPE}, new PortType[] {DatabasePortObject.TYPE});
    }


    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
        if ((m_settings.getQuery() == null) || m_settings.getQuery().isEmpty()) {
            throw new InvalidSettingsException("No query configured");
        }

        DatabaseConnectionPortObjectSpec incomingConnection = (DatabaseConnectionPortObjectSpec)inSpecs[0];
        DatabaseConnectionSettings connSettings = incomingConnection.getConnectionSettings(getCredentialsProvider());

        String sql = FlowVariableResolver.parse(m_settings.getQuery(), this);
        DatabaseQueryConnectionSettings querySettings = new DatabaseQueryConnectionSettings(connSettings, sql);

        final DBReader conn = connSettings.getUtility().getReader(querySettings);

        if (!connSettings.getRetrieveMetadataInConfigure()) {
            return new PortObjectSpec[1];
        }

        try {
            DataTableSpec tableSpec = conn.getDataTableSpec(getCredentialsProvider());

            return new PortObjectSpec[] {new DatabasePortObjectSpec(tableSpec, querySettings)};
        } catch (SQLException ex) {
            Throwable cause = ExceptionUtils.getRootCause(ex);
            if (cause == null) {
                cause = ex;
            }

            throw new InvalidSettingsException(
                "Error while validating SQL query '" + sql + "' : " + cause.getMessage(), ex);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
        exec.setMessage("Retrieving metadata from database");
        DatabaseConnectionPortObject incomingConnection = (DatabaseConnectionPortObject)inObjects[0];
        DatabaseConnectionSettings connSettings = incomingConnection.getConnectionSettings(getCredentialsProvider());

        String sql = FlowVariableResolver.parse(m_settings.getQuery(), this);
        DatabaseQueryConnectionSettings querySettings = new DatabaseQueryConnectionSettings(connSettings, sql);

        DBReader conn = querySettings.getUtility().getReader(querySettings);
        try {
            DataTableSpec tableSpec = conn.getDataTableSpec(getCredentialsProvider());

            return new PortObject[] {new DatabasePortObject(new DatabasePortObjectSpec(tableSpec, querySettings))};
        } catch (SQLException ex) {
            Throwable cause = ExceptionUtils.getRootCause(ex);
            if (cause == null) {
                cause = ex;
            }

            throw new InvalidSettingsException("Error while validating SQL query: " + cause.getMessage(), ex);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec) throws IOException,
        CanceledExecutionException {
        // nothing to do

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec) throws IOException,
        CanceledExecutionException {
        // nothing to do
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        m_settings.saveConnection(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        DatabaseQueryConnectionSettings s = new DatabaseQueryConnectionSettings();
        s.loadValidatedConnection(settings, getCredentialsProvider());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_settings.loadValidatedConnection(settings, getCredentialsProvider());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        // nothing to do
    }
}

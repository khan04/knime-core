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
 *   Jun 5, 2009 (wiswedel): created
 */
package org.knime.core.node.exec;

import java.net.URL;

import org.knime.core.node.port.PortObject;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.AbstractNodeExecutionJobManager;
import org.knime.core.node.workflow.NodeContainer;
import org.knime.core.node.workflow.NodeExecutionJob;
import org.knime.core.node.workflow.execresult.NodeContainerExecutionResult;

/**
 * Job manager that applies a given execution result to a node container. This manager is not visible to the user (not
 * selectable). It is used to load the data of a partially executed metanode in the temporary sandbox workflow, which
 * is then executed, for instance on a cluster.
 *
 * @noreference This class is not intended to be referenced by clients.
 * @author Bernd Wiswedel, KNIME AG, Zurich, Switzerland
 * @since 3.1
 */
final class CopyContentIntoTempFlowNodeExecutionJobManager extends AbstractNodeExecutionJobManager {

    private final NodeContainerExecutionResult m_executionResult;

    /** Create new instance given an execution result.
     * @param executionResult To be applied to the node during the (pseudo-)execution.
     */
    CopyContentIntoTempFlowNodeExecutionJobManager(final NodeContainerExecutionResult executionResult) {
        m_executionResult = CheckUtils.checkArgumentNotNull(executionResult, "Execution result must not be null");
    }

    /** {@inheritDoc} */
    @Override
    public final NodeExecutionJob submitJob(final NodeContainer nc, final PortObject[] data) {
        CopyContentIntoTempFlowNodeExecutionJob result = new CopyContentIntoTempFlowNodeExecutionJob(
            nc, data, m_executionResult);
        result.run();
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public String getID() {
        return CopyContentIntoTempFlowNodeExecutionJobManager.class.getName();
    }

    /** {@inheritDoc} */
    @Override
    public URL getIcon() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public URL getIconForWorkflow() {
        return null;
    }

}

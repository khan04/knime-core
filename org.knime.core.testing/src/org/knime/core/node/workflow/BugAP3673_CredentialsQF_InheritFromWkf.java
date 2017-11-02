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
 * ------------------------------------------------------------------------
 */
package org.knime.core.node.workflow;

import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertThat;

import java.util.Collections;
import java.util.List;

import org.junit.Test;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.util.CheckUtils;
import org.knime.core.node.workflow.WorkflowPersistor.WorkflowLoadResult;

/**
 * Test for workaround described as part of AP-5974. Credentials QF nodes should inherit the info from workflow
 * credentials (for same identifier).
 * @author wiswedel, KNIME AG, Zurich, Switzerland
 */
public class BugAP3673_CredentialsQF_InheritFromWkf extends WorkflowTestCase {

    /** Test credentials that are set during load of workflow. */
    @Test
    public void testExecuteFlow() throws Exception {
        final TestWorkflowLoadHelper lH = new TestWorkflowLoadHelper("some-fixed-password");
        WorkflowLoadResult loadWorkflow = loadWorkflow(getDefaultWorkflowDirectory(), new ExecutionMonitor(), lH);
        setManager(loadWorkflow.getWorkflowManager());
        assertThat("Invalid prompt count - only workflow variables are expected to be prompted",
            lH.getPromptCount(), is(1));
        executeAllAndWait();
        checkState(getManager(), InternalNodeContainerState.EXECUTED);
    }

    /** Test credentials that are set after workflow has been loaded (mimics behavior on the server). */
    @Test
    public void testExecuteFlowThenChangeCredentials() throws Exception {
        // provides no valid credentials, first execution supposed to fail
        WorkflowLoadResult loadWorkflow = loadWorkflow(getDefaultWorkflowDirectory(), new ExecutionMonitor());
        WorkflowManager wfm = loadWorkflow.getWorkflowManager();
        setManager(wfm);
        executeAllAndWait();
        checkState(getManager(), InternalNodeContainerState.IDLE);
        wfm.resetAndConfigureAll();
        wfm.updateCredentials(new Credentials("credentials-input", "some-fixed-username", "some-fixed-password"));
        executeAllAndWait();
        checkState(getManager(), InternalNodeContainerState.EXECUTED);
    }

    private class TestWorkflowLoadHelper extends WorkflowLoadHelper {

        private final String m_password;
        private int m_promptCount;

        /** @param workflowLocation */
        TestWorkflowLoadHelper(final String password) throws Exception {
            super(getDefaultWorkflowDirectory());
            m_password = password;
        }

        /** {@inheritDoc} */
        @Override
        public List<Credentials> loadCredentials(final List<Credentials> credentials) {
            CheckUtils.checkState(credentials.size() == 1, "Expected 1 credentials set, got %d", credentials.size());
            Credentials c = credentials.get(0);
            CheckUtils.checkState(c.getName().equals("credentials-input"), "Wrong identifier: %s", c.getName());
            CheckUtils.checkState(c.getLogin().equals("some-fixed-username"), "Wrong identifier: %s", c.getLogin());
            CheckUtils.checkState(c.getPassword() == null, "Expected null password");

            m_promptCount += 1;
            return Collections.singletonList(new Credentials(c.getName(), c.getLogin(), m_password));
        }

        /**
         * @return the promptCount
         */
        int getPromptCount() {
            return m_promptCount;
        }
    }


}

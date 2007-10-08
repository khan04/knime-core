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
 *  History 
 *      28.06.2007 (Tobias Koetter): created
 */
package org.knime.base.node.groupby;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponent;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentColumnFilter;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.DialogComponentStringSelection;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelFilterString;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * The dialog class of the group by node.
 * 
 * @author Tobias Koetter, University of Konstanz
 */
public class GroupByNodeDialogPane extends DefaultNodeSettingsPane {
    
    private final SettingsModelFilterString m_groupByCols = 
        new SettingsModelFilterString(GroupByNodeModel.CFG_GROUP_BY_COLUMNS);
    
    private final SettingsModelString m_numericColMethod = 
        new SettingsModelString(GroupByNodeModel.CFG_NUMERIC_COL_METHOD, 
                AggregationMethod.getDefaultNumericMethod().getLabel());
    
    private final SettingsModelString m_noneNumericColMethod = 
        new SettingsModelString(GroupByNodeModel.CFG_NONE_NUMERIC_COL_METHOD, 
                AggregationMethod.getDefaultNoneNumericMethod().getLabel());

    private final SettingsModelIntegerBounded m_maxUniqueValues = 
        new SettingsModelIntegerBounded(
                GroupByNodeModel.CFG_MAX_UNIQUE_VALUES, 10000, 1, 
                Integer.MAX_VALUE);
    
    private final SettingsModelBoolean m_moveGroupCols2Front = 
        new SettingsModelBoolean(
                GroupByNodeModel.CFG_MOVE_GROUP_BY_COLS_2_FRONT, false);
    
    private final SettingsModelBoolean m_enableHilite =
        new SettingsModelBoolean(GroupByNodeModel.CFG_ENABLE_HILITE, false);
    
    private final SettingsModelBoolean m_sortInMemory =
        new SettingsModelBoolean(GroupByNodeModel.CFG_SORT_IN_MEMORY, false);
    
    /**Constructor for class GroupByNodeDialogPane.
     */
    GroupByNodeDialogPane() {
        m_noneNumericColMethod.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                m_maxUniqueValues.setEnabled(enableUniqueValuesModel(
                        m_numericColMethod, m_noneNumericColMethod));
            }
        });
        m_numericColMethod.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                m_maxUniqueValues.setEnabled(enableUniqueValuesModel(
                        m_numericColMethod, m_noneNumericColMethod));
            }
        });
        m_maxUniqueValues.setEnabled(enableUniqueValuesModel(
                m_numericColMethod, m_noneNumericColMethod));
        final DialogComponent groupByCols = 
            new DialogComponentColumnFilter(m_groupByCols, 0);
        final DialogComponent numericColMethod = 
            new DialogComponentStringSelection(m_numericColMethod, 
                    "Numerical aggregation method", 
                    AggregationMethod.getNumericalMethodLabels());
        numericColMethod.setToolTipText(
            "This method will be used for all numerical columns");
        final DialogComponent noneNumericColMethod = 
            new DialogComponentStringSelection(m_noneNumericColMethod, 
                    "None numerical aggregation method", 
                    AggregationMethod.getNoneNumericalMethodLabels());
        noneNumericColMethod.setToolTipText(
                "This method will be used for all none numerical columns");
        final DialogComponent maxNoneNumericVals = 
            new DialogComponentNumber(m_maxUniqueValues, 
                    "Maximum unique values per group", 1);
        maxNoneNumericVals.setToolTipText("All groups with more unique values "
                + "will be skipped and replaced by a missing value");
        final DialogComponent enableHilite = new DialogComponentBoolean(
                m_enableHilite, "Enable hiliting");
        final DialogComponent sortInMemory = new DialogComponentBoolean(
                m_sortInMemory, "Sort in memory");
        final DialogComponent moveGroupCols2Front = new DialogComponentBoolean(
                m_moveGroupCols2Front, "Move group column(s) to front");
        
        createNewGroup("Aggregation methods");
        setHorizontalPlacement(true);
        addDialogComponent(noneNumericColMethod);
        addDialogComponent(numericColMethod);
        setHorizontalPlacement(false);
        addDialogComponent(maxNoneNumericVals);
        createNewGroup("Advanced options");
        setHorizontalPlacement(true);
        addDialogComponent(moveGroupCols2Front);
        addDialogComponent(sortInMemory);
        addDialogComponent(enableHilite);
        setHorizontalPlacement(false);
        closeCurrentGroup();
        createNewGroup("Group by column(s)");
        addDialogComponent(groupByCols);
        closeCurrentGroup();
    }

    /**
     * @param numericMethod the numeric method settings model
     * @param noneNumericMethod the none numeric method settings model
     * @return the enable status of the max unique values field
     */
    protected static boolean enableUniqueValuesModel(
            final SettingsModelString numericMethod,
            final SettingsModelString noneNumericMethod) {
        final AggregationMethod numMeth = 
            AggregationMethod.getMethod4SettingsModel(numericMethod);
        final AggregationMethod noneNumMeth = 
            AggregationMethod.getMethod4SettingsModel(noneNumericMethod);
        return numMeth.isUsesLimit() || noneNumMeth.isUsesLimit();
    }
}

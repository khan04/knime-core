/* 
 * -------------------------------------------------------------------
 * This source code, its documentation and all appendant files
 * are protected by copyright law. All rights reserved.
 *
 * Copyright, 2003 - 2009
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
 *   09.02.2005 (georg): created
 */
package org.knime.workbench.ui.wrapper;

import java.awt.EventQueue;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

import javax.swing.JApplet;
import javax.swing.JPanel;
import javax.swing.Timer;

import org.eclipse.swt.SWT;
import org.eclipse.swt.awt.SWT_AWT;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.Composite;

/**
 * Wrapper Composite that uses the SWT_AWT bridge to embed an AWT Panel into a
 * SWT composite.
 * 
 * @author Florian Georg, University of Konstanz
 */
public class Panel2CompositeWrapper extends Composite {
    private Frame m_awtFrame;

    private JPanel m_awtPanel;
    /** see {@link #initX11ErrorHandlerFix()} for details. */
    private static boolean x11ErrorHandlerFixInstalled;

    /**
     * Creates a new wrapper.
     * 
     * @param parent The parent composite
     * @param panel The AWT panel to wrap
     * @param style Style bits, ignored so far
     */
    public Panel2CompositeWrapper(final Composite parent,
            final JPanel panel, final int style) {
        super(parent, style | SWT.EMBEDDED);
        final GridLayout gridLayout = new GridLayout();
        gridLayout.verticalSpacing = 0;
        gridLayout.marginWidth = 0;
        gridLayout.marginHeight = 0;
        gridLayout.horizontalSpacing = 0;
        setLayout(gridLayout);

        m_awtFrame = SWT_AWT.new_Frame(this);
        if (!x11ErrorHandlerFixInstalled && "gtk".equals(SWT.getPlatform())) {
            x11ErrorHandlerFixInstalled = true;
            EventQueue.invokeLater(new Runnable() {
                public void run() {
                    initX11ErrorHandlerFix();
                }
            });
        }
        // use panel as root
        m_awtPanel = panel;
        /* Create another root pane container (JApplet) to enable swing/awt 
         * components and SWT frames/dialogs to work smoothly together:
         * http://www.eclipse.org/articles/article.php?file=Article-Swing-SWT-Integration/index.html
         */
        JApplet wrap = new JApplet();
        wrap.add(m_awtPanel);
        m_awtFrame.add(wrap);
        
        // Pack the frame
        m_awtFrame.pack();
        m_awtFrame.setVisible(true);

    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void dispose() {
        super.dispose();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void checkSubclass() {
    }

    /**
     * @return The wrapped panel, as it can be used within legacy AWT/Swing code
     */
    public JPanel getAwtPanel() {
        return m_awtPanel;
    }
    
    /**
     * Workaround for https://bugs.eclipse.org/bugs/show_bug.cgi?id=171432
     * (comment #15).
     */
    private static void initX11ErrorHandlerFix() {
        assert EventQueue.isDispatchThread();

        try {
            // get XlibWrapper.SetToolkitErrorHandler() and XSetErrorHandler()
            // methods
            Class xlibwrapperClass = Class.forName("sun.awt.X11.XlibWrapper");
            final Method setToolkitErrorHandlerMethod =
                    xlibwrapperClass.getDeclaredMethod(
                            "SetToolkitErrorHandler", null);
            final Method setErrorHandlerMethod =
                    xlibwrapperClass.getDeclaredMethod("XSetErrorHandler",
                            new Class[]{Long.TYPE});
            setToolkitErrorHandlerMethod.setAccessible(true);
            setErrorHandlerMethod.setAccessible(true);

            // get XToolkit.saved_error_handler field
            Class xtoolkitClass = Class.forName("sun.awt.X11.XToolkit");
            final Field savedErrorHandlerField =
                    xtoolkitClass.getDeclaredField("saved_error_handler");
            savedErrorHandlerField.setAccessible(true);

            // determine the current error handler and the value of
            // XLibWrapper.ToolkitErrorHandler
            // (XlibWrapper.SetToolkitErrorHandler() sets the X11 error handler
            // to
            // XLibWrapper.ToolkitErrorHandler and returns the old error
            // handler)
            final Object defaultErrorHandler =
                    setToolkitErrorHandlerMethod.invoke(null, null);
            final Object toolkitErrorHandler =
                    setToolkitErrorHandlerMethod.invoke(null, null);
            setErrorHandlerMethod.invoke(null,
                    new Object[]{defaultErrorHandler});

            // create timer that watches XToolkit.saved_error_handler whether
            // its value is equal
            // to XLibWrapper.ToolkitErrorHandler, which indicates the start of
            // the trouble
            Timer timer = new Timer(200, new ActionListener() {
                public void actionPerformed(final ActionEvent e) {
                    try {
                        Object savedErrorHandler =
                                savedErrorHandlerField.get(null);
                        if (toolkitErrorHandler.equals(savedErrorHandler)) {
                            // Last saved error handler in
                            // XToolkit.WITH_XERROR_HANDLER
                            // is XLibWrapper.ToolkitErrorHandler, which will
                            // cause
                            // the StackOverflowError when the next X11 error
                            // occurs.
                            // Workaround: restore the default error handler.
                            // Also update XToolkit.saved_error_handler so that
                            // this is done only once.
                            setErrorHandlerMethod.invoke(null,
                                    new Object[]{defaultErrorHandler});
                            savedErrorHandlerField.setLong(null,
                                    ((Long)defaultErrorHandler).longValue());
                        }
                    } catch (Exception ex) {
                        // ignore
                    }

                }
            });
            timer.start();
        } catch (Exception ex) {
            // ignore
        }
    }
}

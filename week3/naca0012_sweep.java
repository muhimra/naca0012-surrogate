// =============================================================
// STAR-CCM+ Java Macro — NACA 0012 Parametric Sweep
// Compatible with STAR-CCM+ 2502 (v20.02)
// =============================================================

import star.common.*;
import star.base.report.*;
import star.flow.*;
import java.io.*;

public class naca0012_sweep extends StarMacro {

    // ---- USER CONFIGURATION ----------------------------------------
    static final String INLET_REGION = "Body 2.inlet";
    static final String LIFT_REPORT  = "Lift Coefficient";
    static final String DRAG_REPORT  = "Drag Coefficient";
    static final String OUTPUT_CSV   = "C:/Users/3135581I/OneDrive - University of Glasgow/Personal/CFD SIM STAR/naca0012_sweep.csv";
    static final double NU           = 1.5e-5;   // kinematic viscosity air ~20°C
    static final double CHORD        = 1.0;       // chord length (m)
    static final int    MAX_ITER     = 500;       // iterations per case
    // ----------------------------------------------------------------

    static final double[] ALPHA_DEG = {
        -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    };
    static final double[] RE_VALUES = {
        5e5, 1e6, 3e6
    };

    public void execute() {
        Simulation sim = getActiveSimulation();

        // ---- Find inlet boundary ----
        Boundary inlet = null;
        for (Region region : sim.getRegionManager().getRegions()) {
            try {
                inlet = region.getBoundaryManager().getBoundary(INLET_REGION);
                if (inlet != null) break;
            } catch (Exception e) { /* boundary not in this region */ }
        }
        if (inlet == null) {
            sim.println("[SWEEP] ERROR: Cannot find boundary '" + INLET_REGION + "'");
            return;
        }
        sim.println("[SWEEP] Found boundary: " + inlet.getPresentationName());

        // ---- Get force coefficient reports ----
        ForceCoefficientReport liftReport =
            (ForceCoefficientReport) sim.getReportManager().getReport(LIFT_REPORT);
        ForceCoefficientReport dragReport =
            (ForceCoefficientReport) sim.getReportManager().getReport(DRAG_REPORT);

        // ---- Set solver max steps via stopping criteria ----
        try {
            StepStoppingCriterion stepStop =
                (StepStoppingCriterion) sim.getSolverStoppingCriterionManager()
                    .getSolverStoppingCriterion("Maximum Steps");
            stepStop.setMaximumNumberSteps(MAX_ITER);
            sim.println("[SWEEP] Max iterations set to " + MAX_ITER);
        } catch (Exception e) {
            sim.println("[SWEEP] WARNING: Could not set max steps. (" + e.getMessage() + ")");
        }

        // ---- Open CSV ----
        PrintWriter csv = null;
        try {
            csv = new PrintWriter(new FileWriter(OUTPUT_CSV, false));
        } catch (IOException e) {
            sim.println("[SWEEP] ERROR: Cannot open CSV for writing: " + OUTPUT_CSV);
            return;
        }
        csv.println("alpha_deg,reynolds,velocity_ms,Cl,Cd,ClCd");
        csv.flush();

        int totalRuns = ALPHA_DEG.length * RE_VALUES.length;
        int run = 0;

        for (double re : RE_VALUES) {
            double velocity = re * NU / CHORD;

            for (double alphaDeg : ALPHA_DEG) {
                run++;
                double alphaRad = Math.toRadians(alphaDeg);
                sim.println(String.format(
                    "[SWEEP] %d/%d  alpha=%.1f deg  Re=%.0f  V=%.3f m/s",
                    run, totalRuns, alphaDeg, re, velocity));

                // ---- Update Report Reference Velocities dynamically ----
                liftReport.getReferenceVelocity().setValue(velocity);
                dragReport.getReferenceVelocity().setValue(velocity);

                // ---- Update inlet velocity Magnitude and Flow Direction ----
                star.flow.VelocityMagnitudeProfile velMagProfile = 
                    inlet.getValues().get(star.flow.VelocityMagnitudeProfile.class);
                velMagProfile.getMethod(star.common.ConstantScalarProfileMethod.class)
                    .getQuantity().setValue(velocity);

                star.flow.FlowDirectionProfile flowDirProfile = 
                    inlet.getValues().get(star.flow.FlowDirectionProfile.class);
                flowDirProfile.getMethod(star.common.ConstantVectorProfileMethod.class)
                    .getQuantity().setComponents(
                        Math.cos(alphaRad),
                        Math.sin(alphaRad),
                        0.0);

                // ---- Update force directions ----
                liftReport.getDirection().setComponents(
                    -Math.sin(alphaRad),  Math.cos(alphaRad), 0.0);
                dragReport.getDirection().setComponents(
                     Math.cos(alphaRad),  Math.sin(alphaRad), 0.0);

                // ---- Clear solution fields and run ----
                sim.getSolution().clearSolution(Solution.Clear.Fields);
                sim.getSimulationIterator().run();

                // ---- Extract Cl, Cd ----
                double cl   = liftReport.getReportMonitorValue();
                double cd   = dragReport.getReportMonitorValue();
                double clcd = (Math.abs(cd) > 1e-10) ? cl / cd : 0.0;

                sim.println(String.format(
                    "[SWEEP]   Cl=%.4f  Cd=%.5f  Cl/Cd=%.2f", cl, cd, clcd));

                // ---- Write CSV row ----
                csv.println(String.format(
                    "%.1f,%.0f,%.4f,%.6f,%.6f,%.4f",
                    alphaDeg, re, velocity, cl, cd, clcd));
                csv.flush();
            }
        }

        csv.close();
        sim.println("[SWEEP] Done. Results written to: " + OUTPUT_CSV);
    }
}
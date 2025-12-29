"""Boiler cycling efficiency penalty using EnergyPlus EMS."""

from __future__ import annotations
from eppy.modeleditor import IDF


def add_boiler_efficiency_cycling_ems(
    idf: IDF,
    boiler_name: str = "main boiler",
    efficiency_curve_name: str = "boiler efficiency curve",
    plr_min: float = 0.34,
    k_penalty: float = 0.15,
) -> IDF:
    """
    Add EMS controls for boiler efficiency adjustment based on part-load ratio (PLR)
    and outlet temperature, with cycling penalty below minimum PLR.

    Also tracks boiler cycle count.

    Args:
        idf: The IDF object to modify
        boiler_name: Name of the boiler component in the IDF
        efficiency_curve_name: Name of the efficiency curve to override
        plr_min: Minimum efficient part-load ratio (default 0.34)
        k_penalty: Penalty factor for operation below PLR_min (default 0.15)

    Returns:
        Modified IDF object
    """

    # --- EMS Sensors ---
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:SENSOR",
        Name="BoilerPLR",
        OutputVariable_or_OutputMeter_Index_Key_Name=boiler_name,
        OutputVariable_or_OutputMeter_Name="Boiler Part Load Ratio",
    )

    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:SENSOR",
        Name="BoilerOutletTemp",
        OutputVariable_or_OutputMeter_Index_Key_Name=boiler_name,
        OutputVariable_or_OutputMeter_Name="Boiler Outlet Temperature",
    )

    # --- EMS Actuator ---
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:ACTUATOR",
        Name="BoilerEffCurveOverride",
        Actuated_Component_Unique_Name=efficiency_curve_name,
        Actuated_Component_Type="Curve",
        Actuated_Component_Control_Type="Curve Result",
    )

    # --- EMS Global Variables ---
    # Create a single GlobalVariable object with multiple variables
    glob_vars = idf.newidfobject("ENERGYMANAGEMENTSYSTEM:GLOBALVARIABLE")
    glob_vars.Erl_Variable_1_Name = "BoilerOnPrev"
    glob_vars.Erl_Variable_2_Name = "BoilerCycleCount"

    # --- EMS Programs ---

    # 1. Initialization program
    init_prog = idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAM",
        Name="InitialiseBoilerVariables",
    )
    init_prog.Program_Line_1 = "SET BoilerOnPrev = 0"
    init_prog.Program_Line_2 = "SET BoilerCycleCount = 0"

    # 2. Efficiency adjustment program
    eff_prog = idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAM",
        Name="AdjustBoilerEfficiency",
    )
    eff_prog.Program_Line_1 = "SET PLR = BoilerPLR"
    eff_prog.Program_Line_2 = "SET Tflow = BoilerOutletTemp"
    eff_prog.Program_Line_3 = f"SET PLRmin = {plr_min}"
    eff_prog.Program_Line_4 = f"SET k = {k_penalty}"
    eff_prog.Program_Line_5 = "SET EffBase = 1.24978489 + 0.181034673*PLR - 0.171652899*PLR*PLR - 0.00737313433*Tflow + 0.00003125*Tflow*Tflow + 0.000373134328*PLR*Tflow"
    eff_prog.Program_Line_6 = "IF PLR < PLRmin"
    eff_prog.Program_Line_7 = "SET Penalty = (1 - PLR/PLRmin)*(1 - PLR/PLRmin)"
    eff_prog.Program_Line_8 = "SET EffAdj = EffBase*(1 - k*Penalty)"
    eff_prog.Program_Line_9 = "ELSE"
    eff_prog.Program_Line_10 = "SET EffAdj = EffBase"
    eff_prog.Program_Line_11 = "ENDIF"
    eff_prog.Program_Line_12 = "SET BoilerEffCurveOverride = EffAdj"

    # 3. Cycle counting program
    cycle_prog = idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAM",
        Name="CountBoilerCycles",
    )
    cycle_prog.Program_Line_1 = "IF (BoilerPLR > 0.01)"
    cycle_prog.Program_Line_2 = "IF (BoilerOnPrev == 0)"
    cycle_prog.Program_Line_3 = "SET BoilerCycleCount = BoilerCycleCount + 1"
    cycle_prog.Program_Line_4 = "ENDIF"
    cycle_prog.Program_Line_5 = "SET BoilerOnPrev = 1"
    cycle_prog.Program_Line_6 = "ELSE"
    cycle_prog.Program_Line_7 = "SET BoilerOnPrev = 0"
    cycle_prog.Program_Line_8 = "ENDIF"

    # --- EMS Program Calling Managers ---

    # 1. Call initialization after warmup
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAMCALLINGMANAGER",
        Name="Initialise EMS Variables",
        EnergyPlus_Model_Calling_Point="AfterNewEnvironmentWarmUpIsComplete",
        Program_Name_1="InitialiseBoilerVariables",
    )

    # 2. Call efficiency adjustment after HVAC managers
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAMCALLINGMANAGER",
        Name="Boiler Efficiency Adjustment",
        EnergyPlus_Model_Calling_Point="AfterPredictorAfterHVACManagers",
        Program_Name_1="AdjustBoilerEfficiency",
    )

    # 3. Call cycle counter at end of system timestep
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:PROGRAMCALLINGMANAGER",
        Name="Count Boiler Cycles",
        EnergyPlus_Model_Calling_Point="EndOfSystemTimestepBeforeHVACReporting",
        Program_Name_1="CountBoilerCycles",
    )

    # --- EMS Output Variable ---
    idf.newidfobject(
        "ENERGYMANAGEMENTSYSTEM:OUTPUTVARIABLE",
        Name="Boiler Cycles",
        EMS_Variable_Name="BoilerCycleCount",
        Type_of_Data_in_Variable="Summed",
        Update_Frequency="SystemTimestep",
    )

    return idf

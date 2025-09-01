# boiler_ems.py
from __future__ import annotations
import re
from eppy.modeleditor import IDF

def _san(s: str) -> str:
    """Sanitize a name into an EMS-safe tag (letters, digits, underscore)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def _have(idf: IDF, key: str, name: str) -> bool:
    return any(getattr(o, "Name", "") == name for o in idf.idfobjects.get(key.upper(), []))

def _get(idf: IDF, key: str, name: str):
    for o in idf.idfobjects.get(key.upper(), []):
        if getattr(o, "Name", "") == name:
            return o
    return None

def add_boiler_cycling_penalty(
    idf: IDF,
    boiler_name: str,
    purge_per_cycle_j: float = 12000.0,            # tuning: J of gas per start
    resource_type: str = "NaturalGas",            # meter fuel (e.g., NaturalGas)
    end_use_group: str = "Heating",               # meter end-use group
    end_use_subcat: str = "Boiler Cycling Loss",  # subcategory label in meters
    calling_point: str = "BeginTimestepBeforePredictor",
    add_cycle_counter_output: bool = True,
):
    """
    Adds EMS objects that impose a gas 'pulse' each time the given Boiler:HotWater
    transitions from OFF->ON (PLR 0 -> >0). The pulse is added to the chosen fuel
    meter but does not add heat to the loop (pure loss).

    Returns the modified IDF.
    """
    tag = _san(boiler_name)

    # --- Names (unique per boiler) ---
    sensor_plr        = f"{tag}_BoilPLR"
    g_purge           = f"{tag}_PurgePenaltyGas"
    g_prev_on         = f"{tag}_PrevOn"
    g_cycles          = f"{tag}_CycleCount"
    g_purge_param     = f"{tag}_PurgePerCycleJ"

    metered_var_name  = f"{tag}_Purge_Gas"            # EMS MeteredOutputVariable name
    program_name      = f"{tag}_BoilerPurgeProgram"
    pcm_name          = f"{tag}_BoilerPurge_Manager"

    # --- EMS Sensor: Boiler Part Load Ratio ---
    if not _have(idf, "EnergyManagementSystem:Sensor", sensor_plr):
        idf.newidfobject(
            "EnergyManagementSystem:Sensor".upper(),
            Name=sensor_plr,
            Output_Variable_or_Meter_Index_Key_Name=boiler_name,
            Output_Variable_or_Meter_Name="Boiler Part Load Ratio",
        )

    # --- EMS Globals (state + params) ---
    for gv in (g_purge, g_prev_on, g_cycles, g_purge_param):
        if not _have(idf, "EnergyManagementSystem:GlobalVariable", gv):
            idf.newidfobject(
                "EnergyManagementSystem:GlobalVariable".upper(),
                Name=gv
            )

    # --- Metered output variable (adds fuel to the meter each timestep it is >0) ---
    if not _have(idf, "EnergyManagementSystem:MeteredOutputVariable", metered_var_name):
        idf.newidfobject(
            "EnergyManagementSystem:MeteredOutputVariable".upper(),
            Name=metered_var_name,
            EMS_Variable_Name=g_purge,
            Resource_Type=resource_type,
            Group_Type=end_use_group,
            End_Use_Subcategory=end_use_subcat,
            Units="J",
            Update_Frequency="SystemTimestep",
        )

    # --- Optional: expose cycle count as EMS output variable for reporting ---
    if add_cycle_counter_output and not _have(idf, "EnergyManagementSystem:OutputVariable", f"{tag}_Boiler_Cycles"):
        idf.newidfobject(
            "EnergyManagementSystem:OutputVariable".upper(),
            Name=f"{tag}_Boiler_Cycles",
            EMS_Variable_Name=g_cycles,
            Type_of_Data_in_Variable="Summed",
            Update_Frequency="SystemTimestep",
        )

    # --- EMS Program (one-shot gas penalty on OFF->ON edge) ---
    if not _have(idf, "EnergyManagementSystem:Program", program_name):
        prog = idf.newidfobject(
            "EnergyManagementSystem:Program".upper(),
            Name=program_name,
        )
        # Note: EMS program lines are added as Program_Line_1, Program_Line_2, ...
        lines = [
            f"SET {g_purge_param} = {float(purge_per_cycle_j)}",
            f"SET OnNow = @GreaterThan {sensor_plr} 0.0",
            f"IF (OnNow == 1) && ({g_prev_on} == 0)",
            f"  SET {g_purge} = {g_purge_param}",
            f"  SET {g_cycles} = {g_cycles} + 1",
            f"ELSE",
            f"  SET {g_purge} = 0.0",
            f"ENDIF",
            f"SET {g_prev_on} = OnNow",
        ]
        # write lines into the IDF object fields
        for i, ln in enumerate(lines, start=1):
            setattr(prog, f"Program_Line_{i}", ln)

    # --- Program Calling Manager ---
    if not _have(idf, "EnergyManagementSystem:ProgramCallingManager", pcm_name):
        idf.newidfobject(
            "EnergyManagementSystem:ProgramCallingManager".upper(),
            Name=pcm_name,
            EnergyPlus_Model_Calling_Point=calling_point,
            Program_Name_1=program_name,
        )

    # (Optional) Helpful reporting defaults (comment out if you manage outputs elsewhere)
    # Make sure you have, somewhere in your workflow:
    #   idf.newidfobject("Output:Meter".upper(), Name="NaturalGas:Heating", Reporting_Frequency="Hourly")
    # so you can see the added fuel under the Heating end-use, subcategory “Boiler Cycling Loss”.

    return idf

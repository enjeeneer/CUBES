"""This module implements several window opening models
for naturally ventilated residential buildings"""
import math
import random
from typing import Dict, List

import cubes.rbcs.constants as c
from cubes.rbcs.basecontrol import BaseControl


class CO2ControlledVentilation(BaseControl):
    """this controller opens and closes windows based on co2 thresholds"""

    def __init__(self, open_window_co2, close_window_co2):
        super().__init__()
        self.open_window_co2 = open_window_co2
        self.close_window_co2 = close_window_co2

    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        action_range_dict: Dict[str, List],
    ):
        for zn in c.zone_names:
            if self.open_window_co2 < obs_dict[c.co2_name[zn]]:
                action_dict[c.vent_control_name[zn]] = action_range_dict[
                    c.vent_control_name[zn]
                ][1]
            elif self.close_window_co2 > obs_dict[c.co2_name[zn]]:
                action_dict[c.vent_control_name[zn]] = action_range_dict[
                    c.vent_control_name[zn]
                ][0]
            else:
                if obs_dict[c.vent_name[zn]] > 0:
                    action_dict[c.vent_control_name[zn]] = action_range_dict[
                        c.vent_control_name[zn]
                    ][1]
                else:
                    action_dict[c.vent_control_name[zn]] = action_range_dict[
                        c.vent_control_name[zn]
                    ][0]

        return action_dict


class DOca2014VentilationRate(BaseControl):
    """This controller implements the model published in D'Oca et al. 2014 (Table7):
    Effect of thermostat and window opening occupant behavior models
    on energy use in homes. Build Sim.
    Data is from Denmark.
    neglecting the influence of "sunshine hours" as this is not directly accessible
    """

    def __init__(self, user_type="random"):
        super().__init__()
        if user_type == "random":
            self.user_type = random.choice(["active", "medium", "passive"])
        elif user_type in ["active", "medium", "passive"]:
            self.user_type = user_type
        else:
            print("unknown user type ", user_type)
            print("going with random user type")
            self.user_type = random.choice(["active", "medium", "passive"])

        if self.user_type == "active":
            self.int_open = {"Living": -11.9, "Bedroom": -11.9}
            self.int_close = {"Living": 3.26, "Bedroom": 3.43}
            self.t_in_open = 0.1
            self.t_in_close = -0.08
            self.rh_in_open = {"Bedroom": 0.02, "Living": 0.02}
            self.rh_in_close = {"Bedroom": -0.07, "Living": -0.15}
            self.co2_open = 0
            self.co2_close = 0
            self.t_out_open = {"Bedroom": 0.1, "Living": -0.09}
            self.t_out_close = {"Bedroom": 0, "Living": 0.0}
            self.wind_open = {"Bedroom": 0.34, "Living": 0.34}
            self.wind_close = {"Bedroom": -0.29, "Living": -0.29}
            self.rh_out_open = 0.01
            self.rh_out_close = 0.02
            self.solar_radiation_open = 0
            self.hour_open = [
                0,
                -0.02,
                -0.72,
                -0.43,
                0.97,
                2.48,
                3.0,
                2.81,
                2.49,
                2.12,
                1.69,
                1.85,
                1.67,
                1.46,
                1.51,
                1.59,
                1.93,
                1.9,
                1.38,
                1.0,
                1.2,
                1.8,
                2.15,
                1.84,
            ]
            self.hour_close = [0] * 24

        elif self.user_type == "medium":
            self.int_open = {"Living": -28.75, "Bedroom": -28.75}
            self.int_close = {"Living": -16.16, "Bedroom": -18.27}
            self.t_in_open = 0.15
            self.t_in_close = 0.0
            self.rh_in_open = {"Bedroom": -0.1, "Living": -0.1}
            self.rh_in_close = {"Bedroom": -0.19, "Living": -0.13}
            self.co2_open = 1.4
            self.co2_close = 2.24
            self.t_out_open = {"Bedroom": 0.16, "Living": 0.16}
            self.t_out_close = {"Bedroom": -0.01, "Living": -0.09}
            self.wind_open = {"Bedroom": 0.34, "Living": 0.34}
            self.wind_close = {"Bedroom": 0.47, "Living": 0.47}
            self.rh_out_open = 0.02
            self.rh_out_close = 0.01
            self.solar_radiation_open = 0
            self.hour_open = [
                0.0,
                11.73,
                11.07,
                11.10,
                15.21,
                15.32,
                15.85,
                16.07,
                15.99,
                15.57,
                15.03,
                14.37,
                14.64,
                14.90,
                14.98,
                14.74,
                14.84,
                14.20,
                14.15,
                14.48,
                14.42,
                14.58,
                13.50,
                12.12,
            ]
            self.hour_close = [
                0.0,
                -12.06,
                -0.41,
                -1.10,
                -0.49,
                1.86,
                2.80,
                2.89,
                3.47,
                3.21,
                3.45,
                3.68,
                3.21,
                3.30,
                3.01,
                3.20,
                3.43,
                3.17,
                3.02,
                2.89,
                3.11,
                2.70,
                2.30,
                2.05,
            ]

        else:
            self.int_open = {"Living": -16.50, "Bedroom": -16.10}
            self.int_close = {"Living": -18.56, "Bedroom": -21.19}
            self.t_in_open = 0.14
            self.t_in_close = 0.0
            self.rh_in_open = {"Bedroom": 0.07, "Living": 0.07}
            self.rh_in_close = {"Bedroom": 0.0, "Living": 0.0}
            self.co2_open = 1.01
            self.co2_close = 1.62
            self.t_out_open = {"Bedroom": -0.04, "Living": 0.07}
            self.t_out_close = {"Bedroom": -0.13, "Living": -0.13}
            self.wind_open = {"Bedroom": -0.14, "Living": 0.91}
            self.wind_close = {"Bedroom": 0.0, "Living": 0.0}
            self.rh_out_open = 0.0
            self.rh_out_close = 0.0
            self.solar_radiation_open = 0.19
            self.hour_open = [
                0.0,
                -13.05,
                -13.02,
                -13.00,
                -12.98,
                2.02,
                3.28,
                4.62,
                4.11,
                3.22,
                3.06,
                3.23,
                2.71,
                2.36,
                2.40,
                2.90,
                2.67,
                2.79,
                1.69,
                1.92,
                1.91,
                2.03,
                1.62,
                -13.06,
            ]
            self.hour_close = [0] * 24

    def act(self, obs_dict, action_dict, action_range_dict):

        for zn in c.zone_names:
            if obs_dict[c.occ_name[zn]] == 0:
                action_dict[c.vent_control_name[zn]] = 0

            else:
                rdn = random.random()
                if obs_dict[c.vent_name[zn]] < 1e-2:
                    logit_opening = (
                        self.int_open[zn]
                        + self.t_in_open * obs_dict[c.t_name[zn]]
                        + self.rh_in_open[zn] * obs_dict[c.humidity_name[zn]]
                        + self.co2_open * obs_dict[c.co2_name[zn]]
                        + self.t_out_open[zn] * obs_dict[c.t_out_name]
                        + self.wind_open[zn] * obs_dict[c.windspeed_name]
                        + self.rh_out_open * obs_dict[c.humidity_out_name]
                        + self.solar_radiation_open
                        * (
                            obs_dict[c.direct_solar_radiation_name]
                            + obs_dict[c.diffuse_solar_radiation_name]
                        )
                        + self.hour_open[int(obs_dict[c.hour_name])]
                    )
                    p = 1 / (1 + math.exp(-logit_opening))
                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 1
                    else:
                        action_dict[c.vent_control_name[zn]] = 0

                else:
                    logit_closing = (
                        self.int_close[zn]
                        + self.t_in_close * obs_dict[c.t_name[zn]]
                        + self.rh_in_close[zn] * obs_dict[c.humidity_name[zn]]
                        + self.co2_close * obs_dict[c.co2_name[zn]]
                        + self.t_out_close[zn] * obs_dict[c.t_out_name]
                        + self.wind_close[zn] * obs_dict[c.windspeed_name]
                        + self.rh_out_close * obs_dict[c.humidity_out_name]
                        + self.hour_close[int(obs_dict[c.hour_name])]
                    )
                    p = 1 / (1 + math.exp(-logit_closing))

                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 0
                    else:
                        action_dict[c.vent_control_name[zn]] = 1

        return action_dict


class VentilationRateHaldi2017Denmark(BaseControl):
    """this class implements a window opening model by Haldi et al.
    built on Danish data, published in "Modelling diversity in building occupant
    behaviour: a novel statistical approach",
    Journal of Building Performance Simulation (2017)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()
        self.first = True

    def draw_new_model_numbers(self):
        self.intercept_open = random.gauss(
            mu=-7.36, sigma=0.57
        )  # + random.gauss(mu=0,sigma=2.06)
        self.co2_open = random.gauss(
            mu=0.75e-3, sigma=0.26e-3
        )  # + random.gauss(mu=0,sigma=0.93e-3)
        self.rh_open = random.gauss(
            mu=0.028, sigma=0.01
        )  # + random.gauss(mu=0,sigma=0.032)

        self.intercept_close = random.gauss(
            mu=-0.51, sigma=1.19
        )  # + random.gauss(mu=0,sigma=16.57)
        self.tin_close = random.gauss(
            mu=0.93, sigma=0.29
        )  # + random.gauss(mu=0,sigma=1.16)
        self.tout_close = random.gauss(
            mu=0.57, sigma=0.17
        )  # + random.gauss(mu=0,sigma=0.43)
        self.co2_close = random.gauss(
            mu=7.4e-3, sigma=1.8e-3
        )  # + random.gauss(mu=0,sigma=0.43e-3)
        self.rh_close = random.gauss(
            mu=-0.83, sigma=0.22
        )  # + random.gauss(mu=0,sigma=0.67)

    def act(self, obs_dict, action_dict, action_range_dict):

        for zn in c.zone_names:
            if obs_dict[c.occ_name[zn]] == 0:
                action_dict[c.vent_control_name[zn]] = 0

            else:
                rdn = random.random()
                if obs_dict[c.vent_name[zn]] < 1e-2:
                    logit_opening = (
                        self.intercept_open
                        + self.co2_open * obs_dict[c.co2_name[zn]]
                        + self.rh_open * obs_dict[c.humidity_name[zn]]
                    )
                    p = 1 / (1 + math.exp(-logit_opening))
                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 1
                    else:
                        action_dict[c.vent_control_name[zn]] = 0

                else:
                    logit_closing = (
                        self.intercept_close
                        + self.co2_close * obs_dict[c.co2_name[zn]]
                        + self.tin_close * obs_dict[c.t_name[zn]]
                        + self.tout_close * obs_dict[c.t_out_name]
                        + self.rh_close * obs_dict[c.humidity_name[zn]]
                    )
                    p = 1 / (1 + math.exp(-logit_closing))

                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 0
                    else:
                        action_dict[c.vent_control_name[zn]] = 1

        return action_dict


class VentilationRateRouleau2020(BaseControl):
    """this class implements a window opening model by Rouleau & Gosselin.
    built on Canadian data, published in "Probabilistic window opening model
    considering occupant behavior diversity:
    A data-driven case study of Canadian residential buildings",
    Energy (2020)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()

    def draw_new_model_numbers(self):
        self.omega_op_in = random.gauss(mu=0.059, sigma=0.062)
        self.omega_op_out = random.gauss(mu=0.033, sigma=0.009)

        self.omega_op_const = -27.2 * self.omega_op_in - 98.5 * self.omega_op_out - 1.42
        self.omega_clo_in = 0.3 * self.omega_op_in + 1.07 * self.omega_op_out + 0.04
        self.omega_clo_out = -0.17 * self.omega_op_in + 1.01 * self.omega_op_out
        self.omega_clo_const = (
            -2.18 * self.omega_op_in + 80.7 * self.omega_op_out - 3.37
        )

    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        action_range_dict: Dict[str, List],
    ):

        for zn in c.zone_names:
            if obs_dict[c.occ_name[zn]] == 0:
                action_dict[c.vent_control_name[zn]] = 0

            else:
                rdn = random.random()
                if obs_dict[c.vent_name[zn]] < 1e-2:
                    logit_opening = (
                        self.omega_op_in * obs_dict[c.t_name[zn]]
                        + self.omega_op_out * obs_dict[c.t_out_name]
                        + self.omega_op_const
                    )
                    p = 1 / (1 + math.exp(-logit_opening))
                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 1
                    else:
                        action_dict[c.vent_control_name[zn]] = 0

                else:

                    logit_closing = (
                        self.omega_clo_in * obs_dict[c.t_name[zn]]
                        + self.omega_clo_out * obs_dict[c.t_out_name]
                        + self.omega_clo_const
                    )

                    p = 1 / (1 + math.exp(-logit_closing))

                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 0
                    else:
                        action_dict[c.vent_control_name[zn]] = 1

        return action_dict


class VentilationRateJones2017(BaseControl):
    """this class implements a window opening model by Jones
    built on UK data, published in "Stochastic behavioural models of occupants'
    main bedroom window operation for UK residential buildings",
    Building and Environment (2017)
    """

    def __init__(self,temperature_names:str) -> None:
        super().__init__()
        self.temperature_names = temperature_names
        self.define_model_numbers()

    def get_time_of_day(self, hour):
        # definitions in Jones, 2017
        if 0 <= hour or hour < 6:
            return "morning"
        elif 6 <= hour < 12:
            return "afternoon"
        elif 12 <= hour < 18:
            return "evening"
        elif 18 <= hour < 24:
            return "night"

    def get_season(self, month):
        # definitions in Jones, 2017
        if month == 12 or month <= 2:
            return "winter"
        elif 3 <= month <= 5:
            return "spring"
        elif 6 <= month <= 8:
            return "summer"
        elif 9 <= month <= 11:
            return "autumn"

    def define_model_numbers(self):
        # Jones, 2017, Table 6
        self.intercept_close = {}
        self.intercept_close["spring"] = {
            "morning": -3.727,
            "afternoon": 1.226,
            "evening": -4.3,
            "night": -1.995,
        }
        self.intercept_close["summer"] = {
            "morning": -8.215,
            "afternoon": -5.032,
            "evening": -9.306,
            "night": -14.165,
        }
        self.intercept_close["autumn"] = {
            "morning": -4.017,
            "afternoon": -11.306,
            "evening": -3.049,
            "night": 3.132,
        }
        self.intercept_close["winter"] = {
            "morning": 5.563,
            "afternoon": -14.617,
            "evening": -5.852,
            "night": 43.857,
        }

        self.tin_close = {}
        self.tin_close["spring"] = {
            "morning": -0.102,
            "afternoon": -0.128,
            "evening": -0.276,
            "night": -0.244,
        }
        self.tin_close["summer"] = {
            "morning": 0,
            "afternoon": -0.263,
            "evening": 0.0,
            "night": 0.0,
        }
        self.tin_close["autumn"] = {
            "morning": -0.161,
            "afternoon": 0.268,
            "evening": -0.06,
            "night": -0.667,
        }
        self.tin_close["winter"] = {
            "morning": -0.454,
            "afternoon": 0.337,
            "evening": 0,
            "night": -0.299,
        }

        self.rhin_close = {}
        self.rhin_close["spring"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.rhin_close["summer"] = {
            "morning": -0.038,
            "afternoon": 0,
            "evening": 0,
            "night": 0.216,
        }
        self.rhin_close["autumn"] = {
            "morning": 0.099,
            "afternoon": -0.201,
            "evening": 0.124,
            "night": 0.204,
        }
        self.rhin_close["winter"] = {
            "morning": 0.145,
            "afternoon": -0.189,
            "evening": 0,
            "night": -0.841,
        }

        self.tout_close = {}
        self.tout_close["spring"] = {
            "morning": 0,
            "afternoon": -0.115,
            "evening": 0.239,
            "night": 0,
        }
        self.tout_close["summer"] = {
            "morning": -0.038,
            "afternoon": 0,
            "evening": 0,
            "night": 0.216,
        }
        self.tout_close["autumn"] = {
            "morning": 0.099,
            "afternoon": -0.201,
            "evening": 0.124,
            "night": 0.204,
        }
        self.tout_close["winter"] = {
            "morning": -0.089,
            "afternoon": 0.052,
            "evening": -0.038,
            "night": 0,
        }

        self.rhout_close = {}
        self.rhout_close["spring"] = {
            "morning": 0,
            "afternoon": -0.115,
            "evening": 0.239,
            "night": 0,
        }
        self.rhout_close["summer"] = {
            "morning": 0,
            "afternoon": 0.05,
            "evening": 0,
            "night": 0,
        }
        self.rhout_close["autumn"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.rhout_close["winter"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": -0.508,
        }

        self.windspeed_close = {}
        self.windspeed_close["spring"] = {
            "morning": 0,
            "afternoon": 0.184,
            "evening": 0,
            "night": 0,
        }
        self.windspeed_close["summer"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0.247,
            "night": 0.0,
        }
        self.windspeed_close["autumn"] = {
            "morning": 0,
            "afternoon": 0.146,
            "evening": 0,
            "night": 0.0,
        }
        self.windspeed_close["winter"] = {
            "morning": 0,
            "afternoon": 0.095,
            "evening": 0.199,
            "night": -0.449,
        }

        self.solar_radiation_close = {}
        self.solar_radiation_close["spring"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0.003,
            "night": 0,
        }
        self.solar_radiation_close["summer"] = {
            "morning": -0.001,
            "afternoon": 0.002,
            "evening": 0,
            "night": 0.019,
        }
        self.solar_radiation_close["autumn"] = {
            "morning": -0.003,
            "afternoon": 0.0,
            "evening": -0.392,
            "night": 0.0,
        }
        self.solar_radiation_close["winter"] = {
            "morning": 0.0,
            "afternoon": 0.0,
            "evening": 0,
            "night": 0.0,
        }

        self.rainfall_close = {}
        self.rainfall_close["spring"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0.182,
        }
        self.rainfall_close["summer"] = {
            "morning": 0.0,
            "afternoon": 0.035,
            "evening": 0,
            "night": 0,
        }
        self.rainfall_close["autumn"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0.0,
        }
        self.rainfall_close["winter"] = {
            "morning": 0.067,
            "afternoon": 0.042,
            "evening": 0.0,
            "night": 0.0,
        }

        # Jones, 2017, Table 5
        self.intercept_open = {}
        self.intercept_open["spring"] = {
            "morning": -10.126,
            "afternoon": -3.837,
            "evening": -6.392,
            "night": -2.747,
        }
        self.intercept_open["summer"] = {
            "morning": -7.529,
            "afternoon": -3.358,
            "evening": -8.980,
            "night": -8.580,
        }
        self.intercept_open["autumn"] = {
            "morning": -18.147,
            "afternoon": -17.252,
            "evening": -6.914,
            "night": -34.202,
        }
        self.intercept_open["winter"] = {
            "morning": -6.845,
            "afternoon": -14.406,
            "evening": -3.653,
            "night": -18.420,
        }

        self.tin_open = {}
        self.tin_open["spring"] = {
            "morning": 0.413,
            "afternoon": 0,
            "evening": 0.062,
            "night": -0.043,
        }
        self.tin_open["summer"] = {
            "morning": 0.174,
            "afternoon": -0.155,
            "evening": 0.11,
            "night": -0.148,
        }
        self.tin_open["autumn"] = {
            "morning": 0.498,
            "afternoon": 0.602,
            "evening": 0,
            "night": 0.617,
        }
        self.tin_open["winter"] = {
            "morning": 0,
            "afternoon": 0.363,
            "evening": -0.117,
            "night": -0.709,
        }

        self.rhin_open = {}
        self.rhin_open["spring"] = {
            "morning": 0.053,
            "afternoon": 0,
            "evening": 0.064,
            "night": -0.008,
        }
        self.rhin_open["summer"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.rhin_open["autumn"] = {
            "morning": 0.064,
            "afternoon": 0.087,
            "evening": 0.024,
            "night": 0,
        }
        self.rhin_open["winter"] = {
            "morning": -0.008,
            "afternoon": 0,
            "evening": 0,
            "night": 0.107,
        }

        self.tout_open = {}
        self.tout_open["spring"] = {
            "morning": -0.137,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.tout_open["summer"] = {
            "morning": -0.151,
            "afternoon": 0,
            "evening": 0,
            "night": 0.269,
        }
        self.tout_open["autumn"] = {
            "morning": -0.215,
            "afternoon": -0.250,
            "evening": -0.097,
            "night": 0,
        }
        self.tout_open["winter"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": -0.285,
            "night": 0,
        }

        self.rhout_open = {}
        self.rhout_open["spring"] = {
            "morning": -0.077,
            "afternoon": -0.037,
            "evening": -0.007,
            "night": -0.053,
        }
        self.rhout_open["summer"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.rhout_open["autumn"] = {
            "morning": 0,
            "afternoon": -0.051,
            "evening": 0,
            "night": 0.16,
        }
        self.rhout_open["winter"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": -0.107,
        }

        self.windspeed_open = {}
        self.windspeed_open["spring"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.windspeed_open["summer"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0.164,
            "night": 0.301,
        }
        self.windspeed_open["autumn"] = {
            "morning": 0,
            "afternoon": 0.127,
            "evening": 0,
            "night": 0.149,
        }
        self.windspeed_open["winter"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0.191,
            "night": 0.054,
        }

        self.solar_radiation_open = {}
        self.solar_radiation_open["spring"] = {
            "morning": 0,
            "afternoon": -0.001,
            "evening": -0.009,
            "night": 0,
        }
        self.solar_radiation_open["summer"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0.017,
        }
        self.solar_radiation_open["autumn"] = {
            "morning": 0.002,
            "afternoon": 0.0,
            "evening": 0,
            "night": 0.0,
        }
        self.solar_radiation_open["winter"] = {
            "morning": 0.003,
            "afternoon": 0.004,
            "evening": 0,
            "night": 0.0,
        }

        self.rainfall_open = {}
        self.rainfall_open["spring"] = {
            "morning": 0.039,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        self.rainfall_open["summer"] = {
            "morning": 0.013,
            "afternoon": 0.058,
            "evening": 0,
            "night": 0,
        }
        self.rainfall_open["autumn"] = {
            "morning": 0,
            "afternoon": 0,
            "evening": -0.093,
            "night": 0.0,
        }
        self.rainfall_open["winter"] = {
            "morning": 0.0,
            "afternoon": 0.0,
            "evening": 0.051,
            "night": 0.0,
        }

    def act(self, obs_dict, action_dict, action_range_dict):

        for zn in c.zone_names:
            if obs_dict[c.occ_name[zn]] == 0:
                action_dict[c.vent_control_name[zn]] = 0

            else:
                season = self.get_season(obs_dict[c.month_name])
                time_of_day = self.get_time_of_day(obs_dict[c.hour_name])
                rdn = random.random()
                if obs_dict[c.vent_name[zn]] < 1e-2:
                    logit_opening = (
                        self.intercept_open[season][time_of_day]
                        + self.tin_open[season][time_of_day]
                        * obs_dict[self.temperature_names[zn]]
                        + self.tout_open[season][time_of_day] * obs_dict[c.t_out_name]
                        + self.rhin_open[season][time_of_day]
                        * obs_dict[c.humidity_name[zn]]
                        + self.rhout_open[season][time_of_day]
                        * obs_dict[c.humidity_out_name]
                        + self.windspeed_open[season][time_of_day]
                        * obs_dict[c.windspeed_name]
                        + self.solar_radiation_open[season][time_of_day]
                        * (
                            obs_dict[c.direct_solar_radiation_name]
                            + obs_dict[c.diffuse_solar_radiation_name]
                        )
                        + self.rainfall_open[season][time_of_day]
                        * obs_dict[c.rainfall_name]
                    )
                    p = 1 / (1 + math.exp(-logit_opening))
                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 1
                    else:
                        action_dict[c.vent_control_name[zn]] = 0

                else:

                    logit_closing = (
                        self.intercept_close[season][time_of_day]
                        + self.tin_close[season][time_of_day]
                        * obs_dict[self.temperature_names[zn]]
                        + self.tout_close[season][time_of_day] * obs_dict[c.t_out_name]
                        + self.rhin_close[season][time_of_day]
                        * obs_dict[c.humidity_name[zn]]
                        + self.rhout_close[season][time_of_day]
                        * obs_dict[c.humidity_out_name]
                        + self.windspeed_close[season][time_of_day]
                        * obs_dict[c.windspeed_name]
                        + self.solar_radiation_close[season][time_of_day]
                        * (
                            obs_dict[c.direct_solar_radiation_name]
                            + obs_dict[c.diffuse_solar_radiation_name]
                        )
                        + self.rainfall_close[season][time_of_day]
                        * obs_dict[c.rainfall_name]
                    )

                    p = 1 / (1 + math.exp(-logit_closing))

                    if p > rdn:
                        action_dict[c.vent_control_name[zn]] = 0
                    else:
                        action_dict[c.vent_control_name[zn]] = 1

        return action_dict

"""This module implements several window opening models
for naturally ventilated residential buildings"""
from pyenergyplus.plugin import EnergyPlusPlugin  # type: ignore
import math
import random
from cubes.package.constants import env_files_path


def get_zone_list():
    zone_list = []

    with (
        open(
            (env_files_path + "/list_of_zones.txt"),
            "r",
            encoding="utf-8",
        ) as filehandle
    ):
        for line in filehandle:
            curr_place = line[:-1]
            zone_list.append(curr_place)
    return zone_list


class VentilationRateHaldi2017Denmark(EnergyPlusPlugin):
    """this class implements a window opening model by Haldi et al.
    built on Danish data, published in "Modelling diversity in building occupant
    behaviour: a novel statistical approach",
    Journal of Building Performance Simulation (2017)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()

        self.zone_list = get_zone_list()

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

    def actuate(self, state, actuator_handle, x):
        self.api.exchange.set_actuator_value(state, actuator_handle, x)

    def on_begin_timestep_before_predictor(self, state) -> int:
        if "handles_done" not in self.data:

            self.actuator_ventilation_handles = []
            self.co2_handles = []
            self.tin_handles = []
            self.tout_handles = []
            self.rhin_handles = []
            self.ventrate_handles = []
            self.occupant_count_handles = []

            for zone in self.zone_list:

                self.actuator_ventilation_handles.append(
                    self.api.exchange.get_actuator_handle(
                        state,
                        "Schedule:Constant",
                        "Schedule Value",
                        "Ventilation-Schedule-" + zone,
                    )
                )

                self.api.exchange.request_variable("Zone Air co2 Concentration", zone)
                self.co2_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air co2 Concentration", zone
                    )
                )
                self.api.exchange.request_variable("Zone Mean Air Temperature", zone)
                self.tin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Mean Air Temperature", zone
                    )
                )
                self.api.exchange.request_variable(
                    "Site Outdoor Air Drybulb Temperature", "Environment"
                )
                self.tout_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Site Outdoor Air Drybulb Temperature", "Environment"
                    )
                )
                self.api.exchange.request_variable("Zone Air Relative Humidity", zone)
                self.rhin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air Relative Humidity", zone
                    )
                )
                self.api.exchange.request_variable(
                    "Zone Ventilation Air Change Rate", zone
                )
                self.ventrate_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Ventilation Air Change Rate", zone
                    )
                )
                self.api.exchange.request_variable("Zone People Occupant Count", zone)
                self.occupant_count_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone People Occupant Count", zone
                    )
                )

            if -1 in [
                *self.tin_handles,
                *self.tout_handles,
                *self.rhin_handles,
                *self.ventrate_handles,
                *self.co2_handles,
                *self.actuator_ventilation_handles,
            ]:
                self.api.runtime.issue_severe(
                    state,
                    str(self.tin_handles)
                    + " "
                    + str(self.tout_handles)
                    + " "
                    + str(self.rhin_handles)
                    + " "
                    + str(self.ventrate_handles)
                    + " "
                    + str(self.co2_handles)
                    + " "
                    + str(self.actuator_ventilation_handles),
                )
                return 0

            self.api.runtime.issue_severe(state, "value: " + str(self.intercept_open))

            self.data["handles_done"] = True

        for iz, zone in enumerate(self.zone_list):
            if (
                self.api.exchange.get_variable_value(
                    state, self.occupant_count_handles[iz]
                )
                == 0
            ):
                self.actuate(state, self.actuator_ventilation_handles[iz], 0)
                continue
            rdn = random.random()
            if (
                self.api.exchange.get_variable_value(state, self.ventrate_handles[iz])
                < 0.001
            ):
                co2 = self.api.exchange.get_variable_value(state, self.co2_handles[iz])
                rh = self.api.exchange.get_variable_value(state, self.rhin_handles[iz])

                logit_opening = (
                    self.intercept_open + self.co2_open * co2 + self.rh_open * rh
                )
                p = 1 / (1 + math.exp(-logit_opening))
                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 1)

            else:
                co2 = self.api.exchange.get_variable_value(state, self.co2_handles[iz])
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])
                tout = self.api.exchange.get_variable_value(
                    state, self.tout_handles[iz]
                )
                rhin = self.api.exchange.get_variable_value(
                    state, self.rhin_handles[iz]
                )
                logit_closing = (
                    self.intercept_close
                    + self.co2_close * co2
                    + self.tin_close * tin
                    + self.tout_close * tout
                    + self.rh_close * rhin
                )
                p = 1 / (1 + math.exp(-logit_closing))

                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 0)

        return 0


class VentilationRateRouleau2020(EnergyPlusPlugin):
    """this class implements a window opening model by Rouleau & Gosselin.
    built on Canadian data, published in "Probabilistic window opening model
    considering occupant behavior diversity:
    A data-driven case study of Canadian residential buildings",
    Energy (2020)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()

        self.zone_list = get_zone_list()

    def draw_new_model_numbers(self):
        self.omega_op_in = random.gauss(mu=0.059, sigma=0.062)
        self.omega_op_out = random.gauss(mu=0.033, sigma=0.009)

        self.omega_op_const = -27.2 * self.omega_op_in - 98.5 * self.omega_op_out - 1.42
        self.omega_clo_in = 0.3 * self.omega_op_in + 1.07 * self.omega_op_out + 0.04
        self.omega_clo_out = -0.17 * self.omega_op_in + 1.01 * self.omega_op_out
        self.omega_clo_const = (
            -2.18 * self.omega_op_in + 80.7 * self.omega_op_out - 3.37
        )

    def actuate(self, state, actuator_handle, x):
        self.api.exchange.set_actuator_value(state, actuator_handle, x)

    def on_begin_timestep_before_predictor(self, state) -> int:
        if "handles_done" not in self.data:

            self.actuator_ventilation_handles = []
            self.tin_handles = []
            self.ventrate_handles = []
            self.occupant_count_handles = []

            for zone in self.zone_list:

                self.actuator_ventilation_handles.append(
                    self.api.exchange.get_actuator_handle(
                        state,
                        "Schedule:Constant",
                        "Schedule Value",
                        "Ventilation-Schedule-" + zone,
                    )
                )

                self.tin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Mean Air Temperature", "Zone-1"
                    )
                )
                self.ventrate_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Ventilation Air Change Rate", "Zone-1"
                    )
                )
                self.occupant_count_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone People Occupant Count", zone
                    )
                )

            if -1 in [
                *self.tin_handles,
                *self.ventrate_handles,
                *self.actuator_ventilation_handles,
            ]:
                self.api.runtime.issue_severe(
                    state,
                    str(self.tin_handles)
                    + " "
                    + str(self.ventrate_handles)
                    + " "
                    + str(self.actuator_ventilation_handles),
                )
                return 0

            self.data["handles_done"] = True

        tout = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(
            state,
            self.api.exchange.hour(state),
            self.api.exchange.zone_time_step_number(state),
        )

        for iz, zone in enumerate(self.zone_list):
            if (
                self.api.exchange.get_variable_value(
                    state, self.occupant_count_handles[iz]
                )
                == 0
            ):
                self.actuate(state, self.actuator_ventilation_handles[iz], 0)
                continue

            rdn = random.random()
            if (
                self.api.exchange.get_variable_value(state, self.ventrate_handles[iz])
                < 0.001
            ):
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])

                logit_opening = (
                    self.omega_op_in
                    + tin
                    + self.omega_op_out * tout
                    + self.omega_op_const
                )
                p = 1 / (1 + math.exp(-logit_opening))
                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 1)

            else:
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])

                logit_closing = (
                    self.omega_clo_in
                    + tin
                    + self.omega_clo_out * tout
                    + self.omega_clo_const
                )
                p = 1 / (1 + math.exp(-logit_closing))

                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 0)

        return 0


class VentilationRateJones2017(EnergyPlusPlugin):
    """this class implements a window opening model by Jones
    built on UK data, published in "Stochastic behavioural models of occupants'
    main bedroom window operation for UK residential buildings",
    Building and Environment (2017)
    """

    def __init__(self) -> None:
        super().__init__()
        self.define_model_numbers()

        self.zone_list = get_zone_list()

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

    def actuate(self, state, actuator_handle, x):
        self.api.exchange.set_actuator_value(state, actuator_handle, x)

    def on_begin_timestep_before_predictor(self, state) -> int:
        if "handles_done" not in self.data:

            self.actuator_ventilation_handles = []
            self.tin_handles = []
            self.rhin_handles = []

            self.ventrate_handles = []
            self.occupant_count_handles = []

            for zone in self.zone_list:

                self.actuator_ventilation_handles.append(
                    self.api.exchange.get_actuator_handle(
                        state,
                        "Schedule:Constant",
                        "Schedule Value",
                        "Ventilation-Schedule-" + zone,
                    )
                )

                self.tin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Mean Air Temperature", zone
                    )
                )
                self.rhin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air Relative Humidity", zone
                    )
                )
                self.ventrate_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Ventilation Air Change Rate", zone
                    )
                )
                self.occupant_count_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone People Occupant Count", zone
                    )
                )

            if -1 in [
                *self.occupant_count_handles,
                *self.tin_handles,
                *self.rhin_handles,
                *self.ventrate_handles,
                *self.actuator_ventilation_handles,
            ]:
                self.api.runtime.issue_severe(
                    state,
                    str(self.occupant_count_handles)
                    + " "
                    + str(self.tin_handles)
                    + " "
                    + str(self.rhin_handles)
                    + " "
                    + str(self.ventrate_handles)
                    + " "
                    + str(self.actuator_ventilation_handles),
                )
                return 0

            self.data["handles_done"] = True

        hour = self.api.exchange.hour(state)
        timestep_number = self.api.exchange.zone_time_step_number(state)
        rainfall = self.api.exchange.today_weather_liquid_precipitation_at_time(
            state, hour, timestep_number
        )
        solar_radiation = self.api.exchange.today_weather_beam_solar_at_time(
            state, hour, timestep_number
        ) + self.api.exchange.today_weather_diffuse_solar_at_time(
            state, hour, timestep_number
        )
        tout = self.api.exchange.today_weather_outdoor_dry_bulb_at_time(
            state, hour, timestep_number
        )
        rhout = self.api.exchange.today_weather_outdoor_relative_humidity_at_time(
            state, hour, timestep_number
        )
        windspeed = self.api.exchange.today_weather_wind_speed_at_time(
            state, hour, timestep_number
        )
        season = self.get_season(self.api.exchange.month(state))
        time_of_day = self.get_time_of_day(hour)

        for iz, zone in enumerate(self.zone_list):
            if (
                self.api.exchange.get_variable_value(
                    state, self.occupant_count_handles[iz]
                )
                == 0
            ):
                self.actuate(state, self.actuator_ventilation_handles[iz], 0)
                continue

            rdn = random.random()
            if (
                self.api.exchange.get_variable_value(state, self.ventrate_handles[iz])
                < 0.001
            ):
                rhin = self.api.exchange.get_variable_value(
                    state, self.rhin_handles[iz]
                )
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])

                logit_opening = (
                    self.intercept_open[season][time_of_day]
                    + self.tin_open[season][time_of_day] * tin
                    + self.tout_open[season][time_of_day] * tout
                    + self.rhin_open[season][time_of_day] * rhin
                    + self.rhout_open[season][time_of_day] * rhout
                    + self.windspeed_open[season][time_of_day] * windspeed
                    + self.solar_radiation_open[season][time_of_day] * solar_radiation
                    + self.rainfall_open[season][time_of_day] * rainfall
                )
                p = 1 / (1 + math.exp(-logit_opening))
                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 1)

            else:
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])
                rhin = self.api.exchange.get_variable_value(
                    state, self.rhin_handles[iz]
                )

                logit_closing = (
                    self.intercept_close[season][time_of_day]
                    + self.tin_close[season][time_of_day] * tin
                    + self.tout_close[season][time_of_day] * tout
                    + self.rhin_close[season][time_of_day] * rhin
                    + self.rhout_close[season][time_of_day] * rhout
                    + self.windspeed_close[season][time_of_day] * windspeed
                    + self.solar_radiation_close[season][time_of_day] * solar_radiation
                    + self.rainfall_close[season][time_of_day] * rainfall
                )
                p = 1 / (1 + math.exp(-logit_closing))

                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 0)

        return 0


class VentilationRateAndersen2013Group3Bedroom(EnergyPlusPlugin):
    """this class implements the "group 3 bedroom" window opening model by Andersen
    built on Danish data, published in "Window opening behaviour modelled
    from measurements in Danish dwellings",
    Building and Environment (2013)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()
        self.current_day_of_year = 0
        self.current_sun_hours = 0

        self.zone_list = get_zone_list()

    def get_time_of_day(self, hour):
        if 23 <= hour or hour < 7:
            return "night"
        elif 7 <= hour < 10:
            return "morning"
        elif 10 <= hour < 15:
            return "day"
        elif 15 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 23:
            return "evening"

    def draw_new_model_numbers(self):
        self.intercept_open = {}
        self.intercept_open["night"] = random.gauss(mu=-17.69, sigma=1.1)
        self.intercept_open["morning"] = random.gauss(mu=-15.51, sigma=1.12)
        self.intercept_open["day"] = random.gauss(mu=-17.09, sigma=1.1)
        self.intercept_open["afternoon"] = random.gauss(mu=-18.23, sigma=1.17)
        self.intercept_open["evening"] = random.gauss(mu=-17.13, sigma=1.13)

        self.co2_open = random.gauss(mu=1.75, sigma=0.15)

        self.intercept_close = {}
        self.intercept_close["night"] = random.gauss(mu=-2.68, sigma=3.82)
        self.intercept_close["morning"] = random.gauss(mu=-0.51, sigma=5.86)
        self.intercept_close["day"] = random.gauss(mu=-7.67, sigma=6.17)
        self.intercept_close["afternoon"] = random.gauss(mu=-12.78, sigma=8.16)
        self.intercept_close["evening"] = random.gauss(mu=-13.22, sigma=7.59)

        self.tin_close = {}
        self.tin_close["night"] = random.gauss(mu=0.4, sigma=0.11)
        self.tin_close["morning"] = random.gauss(mu=0.15, sigma=0.12)
        self.tin_close["day"] = random.gauss(mu=0.21, sigma=0.13)
        self.tin_close["afternoon"] = random.gauss(mu=0.7, sigma=0.13)
        self.tin_close["evening"] = random.gauss(mu=0.6, sigma=0.12)

        self.tout_close = {}
        self.tout_close["night"] = random.gauss(mu=0.01, sigma=0.09)
        self.tout_close["morning"] = random.gauss(mu=0.12, sigma=0.09)
        self.tout_close["day"] = random.gauss(mu=-0.13, sigma=0.1)
        self.tout_close["afternoon"] = random.gauss(mu=-0.07, sigma=0.09)
        self.tout_close["evening"] = random.gauss(mu=-0.09, sigma=0.09)

        self.rhin_close = {}
        self.rhin_close["night"] = random.gauss(mu=-0.25, sigma=0.07)
        self.rhin_close["morning"] = random.gauss(mu=-0.16, sigma=0.08)
        self.rhin_close["day"] = random.gauss(mu=0.06, sigma=0.08)
        self.rhin_close["afternoon"] = random.gauss(mu=-0.15, sigma=0.09)
        self.rhin_close["evening"] = random.gauss(mu=-0.07, sigma=0.08)

        self.solar_hours = random.gauss(mu=-0.08, sigma=0.03)

    def update_daily_sun_hours(self, state):
        # a sun hour is defined as 1000 W/m2 solar irradiance for one hour
        dt_per_hour = self.api.exchange.num_time_steps_in_hour(state)
        self.current_sun_hours = 0
        for h in range(0, 24):
            for ts in range(1, dt_per_hour + 1):
                self.current_sun_hours += (
                    (
                        self.api.exchange.today_weather_beam_solar_at_time(state, h, ts)
                        + self.api.exchange.today_weather_diffuse_solar_at_time(
                            state, h, ts
                        )
                    )
                    / dt_per_hour
                    / 1000
                )

    def actuate(self, state, actuator_handle, x):
        self.api.exchange.set_actuator_value(state, actuator_handle, x)

    def on_begin_timestep_before_predictor(self, state) -> int:
        if "handles_done" not in self.data:

            self.actuator_ventilation_handles = []
            self.co2_handles = []
            self.tin_handles = []
            self.tout_handles = []
            self.rhin_handles = []
            self.ventrate_handles = []
            self.occupant_count_handles = []

            for zone in self.zone_list:

                self.actuator_ventilation_handles.append(
                    self.api.exchange.get_actuator_handle(
                        state,
                        "Schedule:Constant",
                        "Schedule Value",
                        "Ventilation-Schedule-" + zone,
                    )
                )

                self.co2_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air co2 Concentration", zone
                    )
                )
                self.tin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Mean Air Temperature", zone
                    )
                )
                self.tout_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Site Outdoor Air Drybulb Temperature", "Environment"
                    )
                )
                self.rhin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air Relative Humidity", zone
                    )
                )
                self.ventrate_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Ventilation Air Change Rate", zone
                    )
                )
                self.occupant_count_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone People Occupant Count", zone
                    )
                )

            if -1 in [
                *self.occupant_count_handles,
                *self.tin_handles,
                *self.tout_handles,
                *self.rhin_handles,
                *self.ventrate_handles,
                *self.co2_handles,
                *self.actuator_ventilation_handles,
            ]:
                self.api.runtime.issue_severe(
                    state,
                    str(self.occupant_count_handles)
                    + " "
                    + str(self.tin_handles)
                    + " "
                    + str(self.tout_handles)
                    + " "
                    + str(self.rhin_handles)
                    + " "
                    + str(self.ventrate_handles)
                    + " "
                    + str(self.co2_handles)
                    + " "
                    + str(self.actuator_ventilation_handles),
                )
                return 0

            self.data["handles_done"] = True

        if self.api.exchange.day_of_year(state) != self.current_day_of_year:
            self.update_daily_sun_hours(state)
            self.current_day_of_year = self.api.exchange.day_of_year(state)

        hour = self.api.exchange.hour(state)

        for iz, zone in enumerate(self.zone_list):
            if (
                self.api.exchange.get_variable_value(
                    state, self.occupant_count_handles[iz]
                )
                == 0
            ):
                self.actuate(state, self.actuator_ventilation_handles[iz], 0)
                continue

            rdn = random.random()
            if (
                self.api.exchange.get_variable_value(state, self.ventrate_handles[iz])
                < 0.001
            ):
                co2 = math.log(
                    self.api.exchange.get_variable_value(state, self.co2_handles[iz])
                )
                logit_opening = (
                    self.intercept_open[self.get_time_of_day(hour)]
                    + self.co2_open * co2
                )
                p = 1 / (1 + math.exp(-logit_opening))
                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 1)

            else:
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])
                tout = self.api.exchange.get_variable_value(
                    state, self.tout_handles[iz]
                )
                rhin = self.api.exchange.get_variable_value(
                    state, self.rhin_handles[iz]
                )

                logit_closing = (
                    self.intercept_close[self.get_time_of_day(hour)]
                    + self.tin_close[self.get_time_of_day(hour)] * tin
                    + self.tout_close[self.get_time_of_day(hour)] * tout
                    + self.rhin_close[self.get_time_of_day(hour)] * rhin
                    + self.solar_hours * self.current_sun_hours
                )
                p = 1 / (1 + math.exp(-logit_closing))

                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 0)

        return 0


class VentilationRateAndersen2013Group3Livingroom(EnergyPlusPlugin):
    """this class implements the "group 3 living room" window opening model by Andersen
    built on Danish data, published in "Window opening behaviour modelled
    from measurements in Danish dwellings",
    Building and Environment (2013)
    """

    def __init__(self) -> None:
        super().__init__()
        self.draw_new_model_numbers()
        self.current_day_of_year = 0
        self.current_sun_hours = 0

        self.zone_list = get_zone_list()

    def get_time_of_day(self, hour):
        if 23 <= hour or hour < 7:
            return "night"
        elif 7 <= hour < 10:
            return "morning"
        elif 10 <= hour < 15:
            return "day"
        elif 15 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 23:
            return "evening"

    def draw_new_model_numbers(self):
        self.intercept_open = {}
        self.intercept_open["night"] = random.gauss(mu=-17.69, sigma=1.1)
        self.intercept_open["morning"] = random.gauss(mu=-15.51, sigma=1.12)
        self.intercept_open["day"] = random.gauss(mu=-17.09, sigma=1.1)
        self.intercept_open["afternoon"] = random.gauss(mu=-18.23, sigma=1.17)
        self.intercept_open["evening"] = random.gauss(mu=-17.13, sigma=1.13)

        self.co2_open = random.gauss(mu=1.75, sigma=0.15)

        self.intercept_close = {}
        self.intercept_close["night"] = random.gauss(mu=14.68, sigma=4.78)
        self.intercept_close["morning"] = random.gauss(mu=16.85, sigma=6.53)
        self.intercept_close["day"] = random.gauss(mu=9.69, sigma=6.81)
        self.intercept_close["afternoon"] = random.gauss(mu=4.57, sigma=8.64)
        self.intercept_close["evening"] = random.gauss(mu=4.13, sigma=8.11)

        self.tin_close = {}
        self.tin_close["night"] = random.gauss(mu=-0.25, sigma=0.12)
        self.tin_close["morning"] = random.gauss(mu=-0.51, sigma=0.18)
        self.tin_close["day"] = random.gauss(mu=-0.45, sigma=0.2)
        self.tin_close["afternoon"] = random.gauss(mu=0.05, sigma=0.18)
        self.tin_close["evening"] = random.gauss(mu=-0.05, sigma=0.24)

        self.tout_close = {}
        self.tout_close["night"] = random.gauss(mu=-0.13, sigma=0.09)
        self.tout_close["morning"] = random.gauss(mu=-0.01, sigma=0.1)
        self.tout_close["day"] = random.gauss(mu=-0.27, sigma=0.09)
        self.tout_close["afternoon"] = random.gauss(mu=-0.2, sigma=0.1)
        self.tout_close["evening"] = random.gauss(mu=-0.22, sigma=0.1)

        self.rhin_close = {}
        self.rhin_close["night"] = random.gauss(mu=-0.25, sigma=0.07)
        self.rhin_close["morning"] = random.gauss(mu=-0.16, sigma=0.08)
        self.rhin_close["day"] = random.gauss(mu=0.06, sigma=0.08)
        self.rhin_close["afternoon"] = random.gauss(mu=-0.15, sigma=0.09)
        self.rhin_close["evening"] = random.gauss(mu=-0.07, sigma=0.08)

        self.solar_hours = random.gauss(mu=-0.08, sigma=0.03)

    def update_daily_sun_hours(self, state):
        # a sun hour is defined as 1000 W/m2 solar irradiance for one hour
        dt_per_hour = self.api.exchange.num_time_steps_in_hour(state)
        self.current_sun_hours = 0
        for h in range(0, 24):
            for ts in range(1, dt_per_hour + 1):
                self.current_sun_hours += (
                    (
                        self.api.exchange.today_weather_beam_solar_at_time(state, h, ts)
                        + self.api.exchange.today_weather_diffuse_solar_at_time(
                            state, h, ts
                        )
                    )
                    / dt_per_hour
                    / 1000
                )

    def actuate(self, state, actuator_handle, x):
        self.api.exchange.set_actuator_value(state, actuator_handle, x)

    def on_begin_timestep_before_predictor(self, state) -> int:
        if "handles_done" not in self.data:

            self.actuator_ventilation_handles = []
            self.co2_handles = []
            self.tin_handles = []
            self.tout_handles = []
            self.rhin_handles = []
            self.ventrate_handles = []
            self.occupant_count_handles = []

            for zone in self.zone_list:

                self.actuator_ventilation_handles.append(
                    self.api.exchange.get_actuator_handle(
                        state,
                        "Schedule:Constant",
                        "Schedule Value",
                        "Ventilation-Schedule-" + zone,
                    )
                )

                self.co2_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air co2 Concentration", zone
                    )
                )
                self.tin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Mean Air Temperature", zone
                    )
                )
                self.tout_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Site Outdoor Air Drybulb Temperature", "Environment"
                    )
                )
                self.rhin_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Air Relative Humidity", zone
                    )
                )
                self.ventrate_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone Ventilation Air Change Rate", zone
                    )
                )
                self.occupant_count_handles.append(
                    self.api.exchange.get_variable_handle(
                        state, "Zone People Occupant Count", zone
                    )
                )

            if -1 in [
                *self.occupant_count_handles,
                *self.tin_handles,
                *self.tout_handles,
                *self.rhin_handles,
                *self.ventrate_handles,
                *self.co2_handles,
                *self.actuator_ventilation_handles,
            ]:
                self.api.runtime.issue_severe(
                    state,
                    str(self.occupant_count_handles)
                    + " "
                    + str(self.tin_handles)
                    + " "
                    + str(self.tout_handles)
                    + " "
                    + str(self.rhin_handles)
                    + " "
                    + str(self.ventrate_handles)
                    + " "
                    + str(self.co2_handles)
                    + " "
                    + str(self.actuator_ventilation_handles),
                )
                return 0

            self.data["handles_done"] = True

        if self.api.exchange.day_of_year(state) != self.current_day_of_year:
            self.update_daily_sun_hours(state)
            self.current_day_of_year = self.api.exchange.day_of_year(state)

        hour = self.api.exchange.hour(state)

        for iz, zone in enumerate(self.zone_list):
            if (
                self.api.exchange.get_variable_value(
                    state, self.occupant_count_handles[iz]
                )
                == 0
            ):
                self.actuate(state, self.actuator_ventilation_handles[iz], 0)
                continue

            rdn = random.random()
            if (
                self.api.exchange.get_variable_value(state, self.ventrate_handles[iz])
                < 0.001
            ):
                co2 = math.log(
                    self.api.exchange.get_variable_value(state, self.co2_handles[iz])
                )
                logit_opening = (
                    self.intercept_open[self.get_time_of_day(hour)]
                    + self.co2_open * co2
                )
                p = 1 / (1 + math.exp(-logit_opening))
                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 1)

            else:
                tin = self.api.exchange.get_variable_value(state, self.tin_handles[iz])
                tout = self.api.exchange.get_variable_value(
                    state, self.tout_handles[iz]
                )
                rhin = self.api.exchange.get_variable_value(
                    state, self.rhin_handles[iz]
                )

                logit_closing = (
                    self.intercept_close[self.get_time_of_day(hour)]
                    + self.tin_close[self.get_time_of_day(hour)] * tin
                    + self.tout_close[self.get_time_of_day(hour)] * tout
                    + self.rhin_close[self.get_time_of_day(hour)] * rhin
                    + self.solar_hours * self.current_sun_hours
                )
                p = 1 / (1 + math.exp(-logit_closing))

                if p > rdn:
                    self.actuate(state, self.actuator_ventilation_handles[iz], 0)

        return 0

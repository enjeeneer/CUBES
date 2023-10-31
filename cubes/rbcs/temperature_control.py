"""This module implements several temperature control strategies"""
import cubes.rbcs.constants as c
from cubes.rbcs.basecontrol import BaseControl
import random
import math
import numpy as np


def draw_set_temp(distribution="Huebner2013_UK"):
    "draw a thermostat set temp from a distribution"
    if distribution == "Huebner2013_UK":
        p = np.array([0.01,0.02,0.01,0.03,0.04,0.08,0.16,0.14,
             0.16,0.16,0.09,0.05,0.03,0.01,0.01])
        return np.random.choice([13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                                p=p/sum(p))
    elif distribution == "EFUS2017_UK":
        p=np.array([0.003,0.002,0.002,0.022,0.021,0.046,0.144,0.096,
                                   0.265,0.151,0.096,0.035,0.020,0.057,0.009,0.003,
                                   0.008,0.002,0.019])
        return np.random.choice([10,13,14,15,16,17,18,19,20,
                                 21,22,23,24,25,26,27,28,29,30],
                                p=p/sum(p))
    else:
        return 20.


once_code_schedule = [(6,23)]
twice_code_schedule = [(6,9),(16,23)]
thrice_code_schedule = [(6,8),(12,14),(18,22)]


def get_onoff_times(sch_name):
    if sch_name == "once_CODE":
        return once_code_schedule
    elif sch_name =="twice_CODE":
        return twice_code_schedule
    elif sch_name =="thrice_CODE":
        return thrice_code_schedule
    elif sch_name == "random":
        index = np.random.choice([1,2,3],
                                p=[0.23,0.58,0.19])
        if index == 1:
            return once_code_schedule
        elif index == 2:
            return twice_code_schedule
        else:
            return thrice_code_schedule



class ConstantTemperature(BaseControl):
    """Controller which sets a constant temperature"""

    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def act(self, obs_dict, action_dict, action_range_dict):
        for zn in c.zone_names:
            action_dict[c.t_control_name[zn]] = self.temp

        return action_dict


class OccupancyControlledTemperature(BaseControl):
    """Controller which sets a temperature depending on zone occupancy"""

    def __init__(self, comfort_temp, setback_temp):
        super().__init__()
        self.comfort_temp = comfort_temp
        self.setback_temp = setback_temp

    def act(self, obs_dict, action_dict, action_range_dict):
        for zn in c.zone_names:
            if obs_dict[c.occ_name[zn]] > 0:
                action_dict[c.t_control_name[zn]] = self.comfort_temp
            else:
                action_dict[c.t_control_name[zn]] = self.setback_temp

        return action_dict


class SwitchOnOFF(BaseControl):
    """Controller which switches heating on and off multiple times a day."""
    def __init__(self, comfort_temp, setback_temp,onoff_times):
        super().__init__()
        if isinstance(comfort_temp, str):
            self.comfort_temp = draw_set_temp(comfort_temp)
        else:
            self.comfort_temp = comfort_temp
        self.setback_temp = setback_temp
        if isinstance(onoff_times,str):
            self.onoff_times = get_onoff_times(onoff_times)
        else:
            self.onoff_times = onoff_times

    def act(self, obs_dict, action_dict, action_range_dict):
        for zn in c.zone_names:
            action_dict[c.t_control_name[zn]] = self.setback_temp
            for on,off in self.onoff_times:
                if on <= obs_dict[c.hour_name] < off:
                    action_dict[c.t_control_name[zn]] = self.comfort_temp

        return action_dict


# class Fabi2013ThermostatControl(BaseControl):
#     """This controller implements the model published in Fabi et al. 2013 (Table6):
#     Influence of occupant’s heating set-point preferenceson indoor
#       environmental quality
#     and heating demandin residential buildings. HVAC&R Research.
#     Data is from Denmark.
#     """
#     def __init__(self,user_type="random"):
#         super().__init__()
#         if user_type == "random":
#             self.user_type = random.choice(["active","medium","passive"])
#         elif user_type in ["active","medium","passive"]:
#             self.user_type = user_type
#         else:
#             print("unknown user type ",user_type)
#             print("going with random user type")
#             self.user_type = random.choice(["active","medium","passive"])

#         if self.user_type == "active":
#             self.rh_in_up = -0.085
#             self.t_out_up = -0.1441

#             self.int_down = -3.514
#             self.solar_rad_down = -0.0194

#         elif self.user_type == "medium":
#             self.int_up = -7.6356
#             self.t_out_up = -0.2284
#             self.wind_up = 0.3699

#             self.int_down = -22.84
#             self.morning_down = 17.68
#             self.noon_down = 16.74
#             self.afternoon_down = 16.26
#             self.evening_down = 16.18

#         else:
#             self.int_up = -9.72
#             self.int_down = -14.28
#             self.solar_rad_down = -1.01

#     def _get_time_of_day(self, hour):
#         if 23 <= hour or hour < 7:
#             return "night"
#         elif 7 <= hour < 10:
#             return "morning"
#         elif 10 <= hour < 15:
#             return "day"
#         elif 15 <= hour < 18:
#             return "afternoon"
#         elif 18 <= hour < 23:
#             return "evening"

#     def _intercept_up_active(self,hour,user_type):

#         if self._get_time_of_day(hour) == "night":
#             return -4.286
#         elif self._get_time_of_day(hour) == "morning":
#             return -0.6264
#         elif self._get_time_of_day(hour) == "day":
#             return -0.839
#         elif self._get_time_of_day(hour) == "afternoon":
#             return -0.8663
#         elif self._get_time_of_day(hour) == "evening":
#             return -2.1435

#     def _intercept_down_medium(self,hour,user_type):
#         if self._get_time_of_day(hour) == "night":
#             return -22.8446
#         elif self._get_time_of_day(hour) == "morning":
#             return -5.1599
#         elif self._get_time_of_day(hour) == "day":
#             return -6.0973
#         elif self._get_time_of_day(hour) == "afternoon":
#             return -6.5805
#         elif self._get_time_of_day(hour) == "evening":
#             return -6.6572


#     def _is_noon(self,hour):
#         return int(self._get_time_of_day(hour) == "noon")
#     def _is_afternoon(self,hour):
#         return int(self._get_time_of_day(hour) == "afternoon")
#     def _is_evening(self,hour):
#         return int(self._get_time_of_day(hour) == "evening")

#     def act(self, obs_dict, action_dict, action_range_dict):
#         for zn in c.zone_names:
#             if obs_dict[c.occ_name[zn]] == 0:
#                 action_dict[c.t_control_name[zn]] = obs_dict[c.t_set_name[zn]]

#             else:
#                 rdn_up = random.random()
#                 rdn_down = random.random()

#                 if self.user_type == "active":
#                     logit_up = (
#                         self.int_up
#                         + self.morning_up * self._is_morning(obs_dict[c.hour_name])
#                         + self.noon_up * self._is_noon(obs_dict[c.hour_name])
#                         + self.afternoon_up
#                           * self._is_afternoon(obs_dict[c.hour_name])
#                         + self.evening_up * self._is_evening(obs_dict[c.hour_name])
#                         + self.rh_in_up * obs_dict[c.humidity_name[zn]]
#                         + self.t_out_up * obs_dict[c.t_out_name]
#                     )

#                     logit_down = (
#                         self.int_down
#                         + self.solar_rad_down * (
#                             obs_dict[c.direct_solar_radiation_name]
#                             + obs_dict[c.diffuse_solar_radiation_name])
#                     )
#                 elif self.user_type == "medium":
#                     logit_up = (
#                         self.int_up
#                         + self.t_out_up * obs_dict[c.t_out_name]
#                         + self.wind_up * obs_dict[c.windspeed_name]
#                     )
#                     logit_down = (
#                         self.int_down
#                         + self.morning_down * self._is_morning(obs_dict[c.hour_name])
#                         + self.noon_down * self._is_noon(obs_dict[c.hour_name])
#                         + self.afternoon_down
#                           * self._is_afternoon(obs_dict[c.hour_name])
#                         + self.evening_down * self._is_evening(obs_dict[c.hour_name])
#                     )
#                 else:
#                     logit_up = (
#                         self.int_up
#                     )
#                     logit_down = (
#                         self.int_down
#                          + self.solar_rad_down * (
#                             obs_dict[c.direct_solar_radiation_name]
#                             + obs_dict[c.diffuse_solar_radiation_name])
#                     )


#                 p_up = 1 / (1 + math.exp(-logit_up))
#                 print("up? ",logit_up,p_up,rdn_up)
#                 if p_up > rdn_up:
#                     action_dict[c.t_control_name[zn]] = obs_dict[c.t_set_name[zn]] + 1
#                 else:
#                     action_dict[c.t_control_name[zn]] = obs_dict[c.t_set_name[zn]]

#                 p_down = 1 / (1 + math.exp(-logit_down))
#                 print("down? ",p_down,rdn_down)
#                 if p_down > rdn_down:
#                     action_dict[c.t_control_name[zn]] -= 1


#         return action_dict


class DOca2014ThermostatControl(BaseControl):
    """This controller implements the model published in D'Oca et al. 2014 (Table6):
    Effect of thermostat and window opening occupant behavior models
    on energy use in homes. Build Sim. (Originally published in Fabi2013!)
    Data is from Denmark.
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
        self.user_type ="medium"
        print(self.user_type)

        self.night_change = 31.3
        self.morning_change = 32.1
        self.day_change = 31.5
        self.afternoon_change = 31.3
        self.evening_change = 27.8
        self.tset_change = -1.28
        self.rh_out_change = -0.0390
        self.rh_in_change = -0.124

        if self.user_type == "active":
            self.night_up = -4.29
            self.morning_up = -0.624
            self.day_up = -0.839
            self.afternoon_up = -0.8663
            self.evening_up = -2.1435
            self.rh_in_up = -0.085
            self.t_out_up = -0.14

            self.int_down = -3.51
            self.solar_rad_down = -0.0194

        elif self.user_type == "medium":
            self.int_up = -7.64
            self.t_out_up = -0.23
            self.wind_up = 0.37

            self.night_down = -22.84
            self.morning_down = -5.1599
            self.day_down = -6.0973
            self.afternoon_down = -6.5805
            self.evening_down = -6.6572

        else:
            self.int_up = -9.72
            self.int_down = -14.28
            self.solar_rad_down = 1.01

    def _get_time_of_day(self, hour):
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

    def _is_night(self, hour):
        return int(self._get_time_of_day(hour) == "night")

    def _is_morning(self, hour):
        return int(self._get_time_of_day(hour) == "morning")

    def _is_day(self, hour):
        return int(self._get_time_of_day(hour) == "day")

    def _is_afternoon(self, hour):
        return int(self._get_time_of_day(hour) == "afternoon")

    def _is_evening(self, hour):
        return int(self._get_time_of_day(hour) == "evening")

    def act(self, obs_dict, action_dict, action_range_dict):
        for zn in c.zone_names:
            print(zn,obs_dict[c.occ_name[zn]])
            if obs_dict[c.occ_name[zn]] == 0:
                action_dict[c.t_control_name[zn]] = obs_dict[c.t_set_name[zn]]

            else:

                if self.user_type == "active":
                    logit_up = (
                        self.night_up * self._is_night(obs_dict[c.hour_name])
                        + self.morning_up * self._is_morning(obs_dict[c.hour_name])
                        + self.day_up * self._is_day(obs_dict[c.hour_name])
                        + self.afternoon_up * self._is_afternoon(obs_dict[c.hour_name])
                        + self.evening_up * self._is_evening(obs_dict[c.hour_name])
                        + self.rh_in_up * obs_dict[c.humidity_name[zn]]
                        + self.t_out_up * obs_dict[c.t_out_name]
                    )

                    logit_down = self.int_down + self.solar_rad_down * (
                        obs_dict[c.direct_solar_radiation_name]
                        + obs_dict[c.diffuse_solar_radiation_name]
                    )
                elif self.user_type == "medium":
                    logit_up = (
                        self.int_up
                        + self.t_out_up * obs_dict[c.t_out_name]
                        + self.wind_up * obs_dict[c.windspeed_name]
                    )
                    logit_down = (
                        self.night_down * self._is_night(obs_dict[c.hour_name])
                        + self.morning_down * self._is_morning(obs_dict[c.hour_name])
                        + self.day_down * self._is_day(obs_dict[c.hour_name])
                        + self.afternoon_down
                        * self._is_afternoon(obs_dict[c.hour_name])
                        + self.evening_down * self._is_evening(obs_dict[c.hour_name])
                    )
                else:
                    logit_up = self.int_up
                    logit_down = self.int_down + self.solar_rad_down * (
                        obs_dict[c.direct_solar_radiation_name]
                        + obs_dict[c.diffuse_solar_radiation_name]
                    )

                change = (
                    self.night_change * self._is_night(obs_dict[c.hour_name])
                    + self.morning_change * self._is_morning(obs_dict[c.hour_name])
                    + self.day_change * self._is_day(obs_dict[c.hour_name])
                    + self.afternoon_change * self._is_afternoon(obs_dict[c.hour_name])
                    + self.evening_change * self._is_evening(obs_dict[c.hour_name])
                    + self.tset_change * obs_dict[c.t_set_name[zn]]
                    + self.rh_in_change * obs_dict[c.humidity_name[zn]]
                    + self.rh_out_change * obs_dict[c.humidity_out_name]
                )

                print("change ", change)
                print("terms ",self.night_change * self._is_night(obs_dict[c.hour_name])
                    ,self.morning_change * self._is_morning(obs_dict[c.hour_name])
                    ,self.day_change * self._is_day(obs_dict[c.hour_name])
                    ,self.afternoon_change * self._is_afternoon(obs_dict[c.hour_name])
                    ,self.evening_change * self._is_evening(obs_dict[c.hour_name])
                    ,self.tset_change * obs_dict[c.t_set_name[zn]]
                    ,self.rh_in_change * obs_dict[c.humidity_name[zn]]
                    ,self.rh_out_change * obs_dict[c.humidity_out_name])

                action_dict[c.t_control_name[zn]] = obs_dict[c.t_set_name[zn]]

                if change > 0:
                    rdn_up = random.random()
                    p_up = 1 / (1 + math.exp(-logit_up))
                    # print(obs_dict[c.t_set_name[zn]],change,logit_up,p_up,rdn_up)
                    if p_up > rdn_up:
                        action_dict[c.t_control_name[zn]] +=  max(0, change)

                else:
                    rdn_down = random.random()
                    p_down = 1 / (1 + math.exp(-logit_down))
                    # print("down? ",p_down,rdn_down)
                    if p_down > rdn_down:
                        action_dict[c.t_control_name[zn]] += min(0, change)

            print(zn,action_dict[c.t_control_name[zn]])

        return action_dict

# coding: utf-8


"""Individual-level trip generation"""
from calendar import week
from datetime import timedelta, datetime
import os
from unicodedata import name
from numpy.core.fromnumeric import choose
import random
import numpy as np
import pandas as pd
from collections import deque
from functools import partial
from collections import Counter
from py2neo import Graph, NodeMatcher
from inspect import Signature, Parameter
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import json


print(os.getcwd())


graph = Graph(
    "http://127.0.0.1:7474", username="neo4j", password="xxxxxx"
) 

def riqi_ftime(df):
    df.insert(1, "riqi", df.ftime.map(lambda x: x.split(" ")[0]))
    df.insert(2, "fftime", df.ftime.map(lambda x: x.split(" ")[1]))
    df.drop("ftime", axis=1, inplace=True)
    df.fftime = df.fftime.map(lambda x: x.split(":")[0] + ":" + x.split(":")[1])
    df.fftime = df.fftime.map(lambda x: timesplit(x))
    return df


def timesplit(input: str, sep=":", gra="15"):
    h, m = input.split(sep)
    m = int(m)
    if gra == "10":
        if m < 10:
            m_str = "00"
        elif m < 20:
            m_str = "10"
        elif m < 30:
            m_str = "20"
        elif m < 40:
            m_str = "30"
        elif m < 50:
            m_str = "40"
        else:
            m_str = "50"
    elif gra == "15":
        if m < 15:
            m_str = "00"
        elif m < 30:
            m_str = "15"
        elif m < 45:
            m_str = "30"
        else:
            m_str = "45"
    return h + ":" + m_str


def transtime_str_to_int(input, sep=":"):
    h, m, _ = input.split(sep)
    output = int(int(h) * 60 + int(m))
    return output
    

def judge_time_slot_for_timemoment(timemoment):
    time = datetime.strptime(timemoment, "%H:%M:%S")

    time_morpeak_start = datetime.strptime("06:30:00", "%H:%M:%S")
    time_morpeak_end = datetime.strptime("08:30:00", "%H:%M:%S")
    time_ant_end = datetime.strptime("13:30:00", "%H:%M:%S")
    time_noon_end = datetime.strptime("14:30:00", "%H:%M:%S")
    time_even_peak_start = datetime.strptime("17:00:00", "%H:%M:%S")
    time_even_peak_end = datetime.strptime("18:30:00", "%H:%M:%S")
    time_even_ping = datetime.strptime("22:30:00", "%H:%M:%S")
    if time >= time_morpeak_start and time < time_morpeak_end:
        shiduan = "Morning_peak"  
    elif time >= time_morpeak_end and time < time_ant_end:
        shiduan = "Morning_flat"
    elif time >= time_ant_end and time < time_noon_end:
        shiduan = "Noon_peak"
    elif time >= time_noon_end and time < time_even_peak_start:
        shiduan = "Noon_flat"
    elif time >= time_even_peak_start and time < time_even_peak_end:
        shiduan = "Evening_peak"
    elif time >= time_even_peak_end and time < time_even_ping:
        shiduan = "Evening_flat"
    else:
        shiduan = "Night"
    return shiduan


    
    

def readTripDadaFromNeo(week = 'weekday'):
    if week == 'weekday': # weekday: five days
        ftime = '2019-08-12'
        ttime = '2019-08-16'
    else: # holiday: two days
        ftime = '2019-08-17'
        ttime = '2019-08-18'
    par_trip_fre = 1.05   
    hphm_new_list, ftime_list, fpark_list, tpark_list = [], [], [], []
    for name in name_list:
        # print(name)
        nodes = graph.run(
            "MATCH (v:Vehicle{name:'%s'}) -[:hasTrip]->(t:Trip),"
            "p1=(t)-[r1:tripFpark]->(n1:Park),"  # fpark
            "p2=(t)-[r2:tripTpark]->(n2:Park), "  # tpark
            "p3=(t)-[r3:tripDate]->(n3:Date) "
            " where n3.name>='%s' and n3.name<='%s'"
            "return t,n1,n2 " % (name,ftime,ttime)
        ).data() 
        for i in nodes:
            hphm_new_list.append(name)
            ftime_list.append(i["t"]["ftime"])
            fpark_list.append(i["n1"]["name"])
            tpark_list.append(i["n2"]["name"])
    output_df = pd.DataFrame(
        {
            "hphm_new": hphm_new_list,
            "ftime": ftime_list,
            "fpark": fpark_list,
            "tpark": tpark_list,
        }
    ) 
    trip_num_max = int(len(output_df)*par_trip_fre) 
    return output_df


def get_time_baseline_from_data(trip_df_for_timeselect:pd.DataFrame):
    trip_df_for_timeselect.insert(
        1, "riqi", trip_df_for_timeselect.ftime.map(lambda x: x.split(" ")[0])
    )
    trip_df_for_timeselect.insert(
        2, "fftime", trip_df_for_timeselect.ftime.map(lambda x: x.split(" ")[1])
    )
    time_slot_counter = Counter(trip_df_for_timeselect.fftime.map(lambda x: judge_time_slot_for_timemoment(x)))
    time_slot_dict = dict(time_slot_counter)
    bg_distribution = {k: v/sum(time_slot_dict.values()) for k,v in time_slot_dict.items()}
    print(bg_distribution)
    trip_df_for_timeselect.drop("ftime", axis=1, inplace=True)
    
    trip_df_for_timeselect.fftime = trip_df_for_timeselect.fftime.map(
        lambda x: transtime_str_to_int(x)
    )
    print(len(trip_df_for_timeselect))
    time_int_counter = Counter(trip_df_for_timeselect.fftime)
    time_int_sum = sum(time_int_counter.values())
    time_int_dict = dict(time_int_counter)
    trip_time_baseline = {
        i: round(1.02 * time_int_dict[i]) if i in time_int_dict else 0
        for i in range(1440)
    }
    return bg_distribution,trip_time_baseline
    

def get_time_baseline_wANDh(week = 'weekday',par_trip_fre = 1.05):
    if week == 'weekday':
        global trip_df 
        global trip_num_max 
        global trip_time_baseline  
        global bg_distribution
        trip_df = readTripDadaFromNeo(week)
        trip_num_max = int(len(trip_df)*par_trip_fre) # The maximum number of trips is set to a multiple of the historical trips, determined by the parameter par_trip_fre
        bg_distribution, trip_time_baseline = get_time_baseline_from_data(trip_df) 
    else:
        global trip_df_holiday
        global trip_num_max_holiday 
        global trip_time_baseline_holiday 
        global bg_distribution_holiday 
        trip_df_holiday = readTripDadaFromNeo('holiday')
        trip_num_max_holiday = int(len(trip_df_holiday)*par_trip_fre)
        bg_distribution_holiday, trip_time_baseline_holiday = get_time_baseline_from_data(trip_df_holiday) 

class Macro_control(object):
    @staticmethod
    def trans_num_to_pro(input: dict):
        num_tol = sum(input.values())
        if num_tol != 0:
            output = {k: v / num_tol for k, v in input.items()}
        else:
            output = {k: 0 for k in input}
        return output

    def __init__(self, bg_distribution,trip_num=100, fluctate_factor=0.05):
        self.trip_num = trip_num  # To prevent initial drastic fluctuations, the given initial assignment value
        self.bg_distribution = bg_distribution
        self.accum_trip = 0
        self.accum_trip_dict = {
            "Morning_peak": 0,
            "Morning_flat": 0,
            "Noon_peak": 0,
            "Noon_flat": 0,
            "Evening_peak": 0,
            "Evening_flat": 0,
            "Night": 0,
        }  
        self.accum_dis_dict = {
            "Morning_peak": 0,
            "Morning_flat": 0,
            "Noon_peak": 0,
            "Noon_flat": 0,
            "Evening_peak": 0,
            "Evening_flat": 0,
            "Night": 0,
        }  
        self.diff_dict_pro = None  
        self.prefer_time_slot = None  
        self.fluctate_factor = fluctate_factor
        self.d_fx = -(1 / self.fluctate_factor)  
        self.accum_trip_dict = {
        k: int(self.trip_num * self.bg_distribution[k])
        for k in self.accum_trip_dict
    }  
        self.cal_time_slot_perfer_pro()

    def cal_distribution(self):
        accum_trip_sum = sum(self.accum_trip_dict.values())
        self.accum_trip = accum_trip_sum 
        self.accum_dis_dict = self.trans_num_to_pro(
            self.accum_trip_dict
        ) 

    def cal_diff_pro(self):
        self.diff_dict_pro = {
            i: (self.accum_dis_dict[i] - self.bg_distribution[i])
            / self.bg_distribution[i]
            for i in self.bg_distribution.keys()
        }  

    def fx(self, x):
        fx = self.d_fx * x + 1
        return fx

    def cal_time_slot_perfer_pro(self):
        self.cal_distribution()
        self.cal_diff_pro()
        self.prefer_time_slot = {
            k: max(0, self.fx(v)) for k, v in self.diff_dict_pro.items()
        }
        for k in set(self.prefer_time_slot.keys()):
            if self.prefer_time_slot[k] == 0:
                del self.prefer_time_slot[k]
        self.prefer_time_slot = self.trans_num_to_pro(self.prefer_time_slot)


class Time_ponit_control(object):
    @staticmethod
    def trans_num_to_pro(input: dict):
        num_tol = sum(input.values())
        if num_tol != 0:
            output = {k: v / num_tol for k, v in input.items()}
        else:
            output = {k: 1/len(input) for k in input}
        return output

    def __init__(
        self,
        dataornot,
        baseline=None,
        trip_num=0,
        func=None,
        popt=None,
        fluctate_factor=0.05,
    ):
        if not dataornot:
            self.trip_num = trip_num
            self.func = func
            self.paramater = popt
            self.fit_trip_sum = int(
                sum([self.func(i, *self.paramater) for i in range(1440)])
            ) 
            self.time_point_num_baseline = {
                i: (self.trip_num * self.func(i, *self.paramater)) / self.fit_trip_sum
                for i in range(1440)
            }
        else:
            self.time_point_num_baseline = baseline
        self.accm_time_point_done = {
            k: 0 for k in self.time_point_num_baseline.keys()
        }  
        
    def time_transtype(self, minutes_int: int):
        return f"{minutes_int // 60}:{minutes_int % 60}:00"

    def time_point_choose(self, start_time, end_time):
        if start_time < end_time:
            x = np.arange(start_time, end_time)
        elif start_time == end_time:
            return self.time_transtype(int(start_time))
        else:
            x = np.hstack((np.arange(start_time, 1440), np.arange(0, end_time)))
        y = {
            i: max(0, (self.time_point_num_baseline[i] - self.accm_time_point_done[i]))
            for i in x
        }  
        y_tol = {
            i: max(0, (self.time_point_num_baseline[i] - self.accm_time_point_done[i]))
            for i in self.time_point_num_baseline.keys()
        }
        if sum(y.values()) > 0:  
            y_list = list(y.keys())
        elif sum(y_tol.values()) > 0:
            y = y_tol
            y_list = list(y.keys())
        else:
            y = {i: self.time_point_num_baseline[i] for i in x}
            y_list = list(y.keys())
        y_pro = self.trans_num_to_pro(y)
        sampler = int(np.random.choice(y_list, p=[y_pro[i] for i in y_list]))
        if sampler not in self.accm_time_point_done:
            self.accm_time_point_done[sampler] = 0
        self.accm_time_point_done[sampler] += 1
        return self.time_transtype(sampler)


class Individual_trip_demand(object):
    """The core class for trip generation
    """

    shiduan_index = {
        "Morning_peak": 1,
        "Morning_flat": 2,
        "Noon_peak": 3,
        "Noon_flat": 4,
        "Evening_peak": 5,
        "Evening_flat": 6,
        "Night": 7,
    }

    shiduan_for_sample = {
        "Morning_peak": (390, 510),
        "Morning_flat": (510, 810),
        "Noon_peak": (810, 870),
        "Noon_flat": (870, 1020),
        "Evening_peak": (1020, 1110),
        "Evening_flat": (1110, 1350),
        "Night": (1350, 390),
    }

    shiduan = (
        "06:30:00",
        "08:30:00",
        "13:30:00",
        "14:30:00",
        "17:00:00",
        "18:30:00",
        "22:30:00",
    )

    @staticmethod
    def trans_num_to_pro(input: dict):
        """ trans number to proportion"""
        num_tol = sum(input.values())
        if num_tol != 0:
            output = {k: v / num_tol for k, v in input.items()}
        else:
            output = {k: 0 for k in input}
        return output

    def __init__(self, vehicle, macro_control_workday, time_control_workday,macro_control_holiday, time_control_holiday, ddl="2019-08-19"):
        self.name = vehicle  # the id of the vehicle individual being generated
        self.macro_control_workday = macro_control_workday
        self.time_control_workday = time_control_workday
        self.macro_control_holiday = macro_control_holiday
        self.time_control_holiday = time_control_holiday
        self.macro_control = None
        self.time_control = None
        
        self.present_time = datetime(2019, 8, 12, 6, 30, 0)  # Time initialization
        self.time_nyr = datetime.strftime(
            self.present_time, "%Y-%m-%d"
        )  
        self.time_sfm = datetime.strftime(
            self.present_time, "%H:%M:%S"
        ) 
        self.week = None 
        self.week_switch = 0 
        self.present_shiduan = None  # related to present_time, e.g., Morning_peak/Evening_peak
        self.week_shiduan = None  
        # Variables related to trip frequency
        self.trip_num_tol = None  
        self.trip_sum_per_day = None  # daily trip frequency dtype:float
        self.trip_num_the_day = None  
        self.trip_num_done = 0  
        self.trip_num_need = 0  
        self.trip_num_accm = 0 
        self.ddl = ddl  
        # Variables related to trip time determination
        self.macro_time_slot_dict = macro_control.prefer_time_slot  
        self.trip_bg_time_slot_whole = {}  
        self.trip_time_slot_thePark = {}  
        self.trip_time_slot_can_choose = {}  
        self.trip_time_slot_latter = set()  
        self.time_slot_choosen = None  
        self.accm_time_slot_num = {}  
        # Variables related to trip destination determination
        self.present_park = None  
        self.origin_park = None  
        self.main_park = None  
        self.destination_park = None  
        self.bg_tripDpark = {}  
        self.bg_tripOpark = {}  
        self.accm_tripOpark = {}  
        self.accm_tripOpark_pro = {} 
        self.accm_tripDpark = {} 
        self.accm_tripDpark_pro = {} 
        self.trip_tpark_thePark = {}  
        self.new_Fpark = None 
        #Variables to enhance robustness
        self.trap_deque = deque(maxlen=4)  
        self.maxstay_time = 0  
        self.maxstay_dict = {}  
        self.time_sum_dict = {}  

        self.judge_week_time_slot(self.present_time, present=1)
        self.switch_controlor() 
        self.instance_init()
        self.traveltime = 0
        self.ss = 1

    def instance_init(self):
        """初始化：self.trip_bg_time_slot_whole,self.bg_tripDpark,self.main_park
                  self.bg_tripOpark,self.trip_num_tol,self.trip_sum_per_day
                  self.present_park,self.bg_tripFroad,self.bg_tripDroad"""
        nodes = graph.run(
            "MATCH (v:Vehicle{name:'%s'}) -[:hasTrip]->(t:Trip),"
            "p2=(t)-[r2:tripTimeseg]->(n2),"
            "p3=(t)-[r3:tripFpark]->(n3:Park),"
            "p6=(t)-[r6:pass]->(n6:Road),"  # froad
            "p4=(t)-[r4:tripTpark]->(n4:Park), "
            "p7=(t)-[r7:pass]->(n7:Road), "  # troad
            "p5=(t)-[r5:tripDate]->(n5:Date), "
            "p1=(t)-[r1:tripWeek]->(n1:Week) "
            " where n1.name='%s' and r6.order='0' and r7.mark='end' and "
            "n5.name>='2019-08-12' and n5.name<='2019-08-18'"
            "return n2,n3,n4,n6,n7 " % (self.name,self.week)
        ).data()  
        self.trip_num_tol = len(nodes) 
        if self.trip_num_tol == 0:
           self.trip_sum_per_day = 0
           return 'done'
        if self.week == 'weekday':
            day_num = 5 
        else:
            day_num = 2
        self.trip_sum_per_day = round(self.trip_num_tol / day_num, 1) 
        self.cal_the_day_trip_num()
        
        trip_time_slot_list = []
        fpark_list = []
        tpark_list = []
        froad_list = []
        troad_list = []
        for i in nodes:
            trip_time_slot_list.append(i["n2"]["name"])
            fpark_list.append(i["n3"]["name"])
            tpark_list.append(i["n4"]["name"])
            froad_list.append(i["n6"]["name"])
            troad_list.append(i["n7"]["name"])
        counter_trip_time_slot = Counter(trip_time_slot_list)
        counter_fpark = Counter(fpark_list)
        counter_tpark = Counter(tpark_list)
        counter_froad = Counter(froad_list)
        counter_troad = Counter(troad_list)
        tuple_fpark = sorted(
            counter_fpark.items(), key=lambda x: x[1], reverse=True
        )  
        self.trip_bg_time_slot_whole = dict(
            sorted(counter_trip_time_slot.items(), key=lambda x: x[1], reverse=True)
        ) 
        self.bg_tripDpark = dict(
            sorted(counter_tpark.items(), key=lambda x: x[1], reverse=True)
        ) 
        self.bg_tripOpark = dict(tuple_fpark)
        self.bg_tripOroad = dict(
            sorted(counter_froad.items(), key=lambda x: x[1], reverse=True)
        )
        self.bg_tripDroad = dict(
            sorted(counter_troad.items(), key=lambda x: x[1], reverse=True)
        )
        self.main_park = tuple_fpark[0][0]  
        self.present_park = tuple_fpark[0][0]  

    def judge_week_time_slot(self, timemoment, present=0):  
        week = timemoment.weekday()
        if week == 5 or week == 6:
            weekday = "holiday"
        else:
            weekday = "weekday"

        time = datetime.strftime(timemoment, "%H:%M:%S")  
        time = datetime.strptime(time, "%H:%M:%S")

        time_morpeak_start = datetime.strptime("06:30:00", "%H:%M:%S")
        time_morpeak_end = datetime.strptime("08:30:00", "%H:%M:%S")
        time_ant_end = datetime.strptime("13:30:00", "%H:%M:%S")
        time_noon_end = datetime.strptime("14:30:00", "%H:%M:%S")
        time_even_peak_start = datetime.strptime("17:00:00", "%H:%M:%S")
        time_even_peak_end = datetime.strptime("18:30:00", "%H:%M:%S")
        time_even_ping = datetime.strptime("22:30:00", "%H:%M:%S")
        
        if time >= time_morpeak_start and time < time_morpeak_end:
            shiduan = "Morning_peak"  
        elif time >= time_morpeak_end and time < time_ant_end:
            shiduan = "Morning_flat"
        elif time >= time_ant_end and time < time_noon_end:
            shiduan = "Noon_peak"
        elif time >= time_noon_end and time < time_even_peak_start:
            shiduan = "Noon_flat"
        elif time >= time_even_peak_start and time < time_even_peak_end:
            shiduan = "Evening_peak"
        elif time >= time_even_peak_end and time < time_even_ping:
            shiduan = "Evening_flat"
        else:
            shiduan = "Night"
        if present == 1: 
            self.week = weekday
            self.present_shiduan = shiduan
            self.week_shiduan = f"{weekday} {shiduan}"  
        else:
            return f"{weekday}"  

    def cal_trip_time_slotAndTaprk_thePark(self):
        time_slot_Thepark_list = []
        tpark_Thepark_list = []
        node = graph.run(
            "MATCH (v:Vehicle{name:'%s'}) -[:hasTrip]->(t:Trip),p1=(t)-[r1:tripWeek]->(n1),"
            "p2=(t)-[r2:tripTimeseg]->(n2),"
            "p3=(t)-[r3:tripFpark]->(n3:Park),"
            "p4=(t)-[r4:tripTpark]->(n4:Park), "
            "p5=(t)-[r5:tripDate]->(n5:Date) "
            " where n3.name='%s' and n1.name='%s' \
            and n5.name>='2019-08-01' and n5.name<'2019-09-03' " 
            " return n1,n2,n4 " % (self.name, self.present_park,self.week)
        ).data()
        for i in node:
            week = i["n1"]["name"]
            time_slot = i["n2"]["name"]
            tpark_Thepark = i["n4"]["name"]
            tpark_Thepark_list.append(tpark_Thepark)
            time_slot_Thepark_list.append(time_slot)
        """ Both of the following dictionaries may be empty, which means that no trip 
         has been made using the current cell as the starting point, caused by the 
         discontinuity of the trajectory reconstruction"""
        self.trip_tpark_thePark = dict(
            Counter(tpark_Thepark_list)
        ) 
        self.trip_time_slot_thePark = dict(
            Counter(time_slot_Thepark_list)
        )  
        self.trip_time_slot_thePark_pro = self.trans_num_to_pro(
            self.trip_time_slot_thePark
        )
        if len(self.trip_time_slot_thePark) == 0: 
            self.select_new_Opark() 
            self.present_park = self.new_Fpark
            self.cal_time_slotAndTaprk_thePark_for_newpark()  
        if self.present_park not in self.accm_tripOpark:
            self.accm_tripOpark[self.present_park] = 0
        self.accm_tripOpark[self.present_park] += 1

    def cal_time_slotAndTaprk_thePark_for_newpark(self):
        time_slot_Thepark_list = []
        tpark_Thepark_list = []
        node = graph.run(
            "MATCH (v:Vehicle{name:'%s'}) -[:hasTrip]->(t:Trip),p1=(t)-[r1:tripWeek]->(n1),"
            "p2=(t)-[r2:tripTimeseg]->(n2),"
            "p3=(t)-[r3:tripFpark]->(n3:Park),"
            "p4=(t)-[r4:tripTpark]->(n4:Park), "
            "p5=(t)-[r5:tripDate]->(n5:Date) "
            " where n3.name='%s' and n1.name='%s' \
            and n5.name>='2019-08-01' and n5.name<'2019-09-03' "  
            " return n1,n2,n4 " % (self.name, self.present_park,self.week)
        ).data()
        for i in node:
            week = i["n1"]["name"]
            time_slot = i["n2"]["name"]
            tpark_Thepark = i["n4"]["name"]
            tpark_Thepark_list.append(tpark_Thepark)
            time_slot_Thepark_list.append(time_slot)
        """ Both of the following dictionaries may be empty, which means that no trip 
         has been made using the current cell as the starting point, caused by the 
         discontinuity of the trajectory reconstruction"""
        self.trip_tpark_thePark = dict(
            Counter(tpark_Thepark_list)
        )  
        self.trip_time_slot_thePark = dict(
            Counter(time_slot_Thepark_list)
        )  
        self.trip_time_slot_thePark_pro = self.trans_num_to_pro(
            self.trip_time_slot_thePark
        ) 
    
    def switch_controlor(self):
        if self.week ==  'weekday':
            self.macro_control = self.macro_control_workday
            self.time_control = self.time_control_workday
        else:
            self.macro_control = self.macro_control_holiday
            self.time_control = self.time_control_holiday     

    def cal_the_day_trip_num(self):
        """switch to a new day, update self.trip_num_the_day"""
        fraction_part = self.trip_sum_per_day % 1
        int_part = int(self.trip_sum_per_day)
        plus_one_or_not = np.random.choice([1, 0], p=[fraction_part, 1 - fraction_part])
        self.trip_num_the_day = int_part + plus_one_or_not  
        self.trip_num_need = self.trip_num_the_day 

    
    def cal_time_slot_can_choose(self):
        """determine time slot can be chosen , update self.time_slot_choosen"""
        if self.trip_num_need > 0:
            self.judge_week_time_slot(self.present_time, present=1)
            self.trip_time_slot_latter = set(
            [
                i
                for i in self.shiduan_index
                if self.shiduan_index[i] >= self.shiduan_index[self.present_shiduan]
            ]
        )
        else:
            time_nyr = datetime.strptime(self.time_nyr, "%Y-%m-%d") + timedelta(days=1)
            week_the_day = time_nyr.weekday()
            self.time_nyr = datetime.strftime(time_nyr, "%Y-%m-%d")  # switch to a new day.
            if self.time_nyr >= self.ddl:
                return 'done' # The generation of this individual is complete
            else:
                
                if week_the_day >= 5 and self.week_switch == 0:
                    # switch to holiday
                    self.week = 'holiday'
                    self.switch_controlor() # change the controller
                    self.instance_init() # Initialization
                    self.week_switch = 1
                else:    
                    self.cal_the_day_trip_num()
                
                self.trip_time_slot_latter = self.shiduan_index.keys() 
                
        if self.trip_num_need>0:
            self.ss = 1
            macro_whole_time_slot_can_choose = (
                self.macro_time_slot_dict.keys() & self.trip_bg_time_slot_whole.keys()
            )
            latter_macro_whole_time_slot_can_choose = (
                self.trip_time_slot_latter & macro_whole_time_slot_can_choose
            )
            self.cal_trip_time_slotAndTaprk_thePark()  
            latter_macro_time_slot_can_choose = (
                self.macro_time_slot_dict.keys() & self.trip_time_slot_latter
            )
            locked_time_solt_num = min(3,self.trip_num_need - 1)  
            pop_latter_macro_time_slot_can_choose_list = sorted(
                list(latter_macro_time_slot_can_choose), key=lambda x: self.shiduan_index[x]
            )
            if len(pop_latter_macro_time_slot_can_choose_list) > locked_time_solt_num:
                for i in range(locked_time_solt_num):
                    pop_latter_macro_time_slot_can_choose_list.pop() # pop locked time slots
            else:
                pop_latter_macro_time_slot_can_choose_list = []
                
            pop_latter_macro_whole_time_slot_can_choose_list = sorted(
                list(latter_macro_whole_time_slot_can_choose),
                key=lambda x: self.shiduan_index[x],
            )
            if len(pop_latter_macro_whole_time_slot_can_choose_list) > locked_time_solt_num:
                for i in range(locked_time_solt_num):
                    pop_latter_macro_whole_time_slot_can_choose_list.pop()
            else:
                pop_latter_macro_whole_time_slot_can_choose_list = []

            if (
                len(
                    set(pop_latter_macro_time_slot_can_choose_list)
                    & self.trip_time_slot_thePark.keys()
                )
                >= 2
            ):
            
                if (
                    len(
                        set(pop_latter_macro_whole_time_slot_can_choose_list)
                        & self.trip_time_slot_thePark.keys()
                    )
                    >= 2
                ):
                    self.trip_time_slot_can_choose = (
                        set(pop_latter_macro_whole_time_slot_can_choose_list)
                        & self.trip_time_slot_thePark.keys()
                    )
                else:
                    self.trip_time_slot_can_choose = (
                        set(pop_latter_macro_time_slot_can_choose_list)
                        & self.trip_time_slot_thePark.keys()
                    )
                time_slot_can_choose_list = list(self.trip_time_slot_can_choose)
                time_slot_can_choose_pro = self.trans_num_to_pro(
                    {
                        k: self.macro_time_slot_dict[k] * self.trip_time_slot_thePark_pro[k]
                        for k in time_slot_can_choose_list
                    }
                )
                # Determine the time period
                self.time_slot_choosen = np.random.choice(
                    time_slot_can_choose_list,
                    p=[time_slot_can_choose_pro[i] for i in time_slot_can_choose_list],
                )
                self.history_OT_choose_tpark()
            elif len(set(pop_latter_macro_whole_time_slot_can_choose_list)) >= 2:
                self.trip_time_slot_can_choose = set(
                    pop_latter_macro_whole_time_slot_can_choose_list
                )
                time_slot_can_choose_list = pop_latter_macro_whole_time_slot_can_choose_list
                time_slot_can_choose_pro = self.trans_num_to_pro(
                    {
                        k: self.macro_time_slot_dict[k] * self.trip_bg_time_slot_whole[k]
                        for k in time_slot_can_choose_list
                    }
                )
                # Determine the time period
                self.time_slot_choosen = np.random.choice(
                    time_slot_can_choose_list,
                    p=[time_slot_can_choose_pro[i] for i in time_slot_can_choose_list],
                )
                # have to select a new park
                self.non_history_OT_choose_tpark()
            # trip time logic constrain
            elif len(set(pop_latter_macro_time_slot_can_choose_list)) >= 2:
                time_slot_can_choose_list = pop_latter_macro_time_slot_can_choose_list
                self.trip_time_slot_can_choose = set(time_slot_can_choose_list)
    
                time_slot_can_choose_pro = self.trans_num_to_pro(
                    {k: self.macro_time_slot_dict[k] for k in time_slot_can_choose_list}
                )
  
                self.time_slot_choosen = np.random.choice(
                    time_slot_can_choose_list,
                    p=[time_slot_can_choose_pro[i] for i in time_slot_can_choose_list],
                )
        
                self.non_history_OT_choose_tpark()
            else:  
                time_slot_can_choose_list = list(self.macro_time_slot_dict.keys())
                self.trip_time_slot_can_choose = set(time_slot_can_choose_list)
                time_slot_can_choose_pro = self.trans_num_to_pro(
                    {k: self.macro_time_slot_dict[k] for k in time_slot_can_choose_list}
                )
                self.time_slot_choosen = np.random.choice(
                    time_slot_can_choose_list,
                    p=[time_slot_can_choose_pro[i] for i in time_slot_can_choose_list],
                )
                self.non_history_OT_choose_tpark()
            self.trip_num_need -= 1  
            self.trip_num_accm += 1
            self.macro_control.accum_trip_dict[self.time_slot_choosen] += 1
            if self.time_slot_choosen not in self.accm_time_slot_num:
                self.accm_time_slot_num[self.time_slot_choosen] = 0
            self.accm_time_slot_num[self.time_slot_choosen] += 1
        else:
            self.ss=0

    def cal_specific_time(self):
        """The function to determine the trip time period"""
        time_sfm = datetime.strftime(self.present_time, "%H:%M:%S")  
        self.time_sfm = time_sfm
        if transtime_str_to_int(time_sfm) < 390 and self.time_slot_choosen == 'Night':
            sample_start_time = transtime_str_to_int(time_sfm)
        else:
            sample_start_time = max(
                transtime_str_to_int(time_sfm),
                self.shiduan_for_sample[self.time_slot_choosen][0],
            )
        chufa_shijian = self.time_control.time_point_choose(
            sample_start_time, self.shiduan_for_sample[self.time_slot_choosen][1]
        )
        ftime_specific_time = datetime.strptime(
            f"{self.time_nyr} {chufa_shijian}", "%Y-%m-%d %H:%M:%S"
        )
        if  chufa_shijian >= '00:00:00':
            self.ftime_specific_time = datetime.strptime(
                f"{self.time_nyr} {chufa_shijian}", "%Y-%m-%d %H:%M:%S"
            )  
        else:
            self.ftime_specific_time = datetime.strptime(
                f"{self.time_nyr} {chufa_shijian}", "%Y-%m-%d %H:%M:%S"
            ) + timedelta(days=1)
        self.present_time = self.ftime_specific_time 

    def history_OT_choose_tpark(self):
        nodes = graph.run(
            "MATCH (v:Vehicle{name:'%s'}) -[:hasTrip]->(t:Trip),"
            "p2=(t)-[r2:tripTimeseg]->(n2),"
            "p4=(t)-[r4:tripTpark]->(n4:Park), "
            "p5=(t)-[r5:tripDate]->(n5:Date), "
            "p6=(t)-[r6:tripWeek]->(n6:Week) "
            " where n5.name>='2019-08-01' and n5.name<'2019-09-03' " 
            " and n2.name='%s' and n6.name='%s' return n4 "
            % (self.name, self.time_slot_choosen,self.week)
        ).data()
        if len(nodes) != 0:
            tpark_can_choose_list = []
            for i in nodes:
                tpark_can_choose_list.append(i["n4"]["name"])
            tpark_can_choose_pro = self.trans_num_to_pro(
                dict(Counter(tpark_can_choose_list))
            )
            tpark_can_choose_list = list(tpark_can_choose_pro.keys())
        else:
            tpark_can_choose_list = list(self.trip_tpark_thePark.keys()) 
            tpark_can_choose_pro = self.trans_num_to_pro(
                self.trip_tpark_thePark
            ) 
        self.destination_park = np.random.choice(
            tpark_can_choose_list,
            p=[tpark_can_choose_pro[i] for i in tpark_can_choose_list],
        )
        if self.destination_park not in self.accm_tripDpark:
            self.accm_tripDpark[self.destination_park] = 0
        self.accm_tripDpark[self.destination_park] += 1

    def non_history_OT_choose_tpark(self):
        self.bg_tripDpark_pro = self.trans_num_to_pro(self.bg_tripDpark)
        self.accm_tripDpark_pro = self.trans_num_to_pro(self.accm_tripDpark)
        tpark_new_can_choose = {
            k: (
                max(0, v - self.accm_tripDpark_pro[k])
                if k in self.accm_tripDpark_pro
                else v
            )
            for k, v in self.bg_tripDpark_pro.items()
        }
        if sum(tpark_new_can_choose.values()) != 0:
            tpark_new_can_choose = self.trans_num_to_pro(tpark_new_can_choose)
        else:
            tpark_new_can_choose = self.bg_tripDpark_pro
        tpark_new_can_choose_list = list(tpark_new_can_choose.keys())
        self.destination_park = np.random.choice(
            tpark_new_can_choose_list,
            p=[tpark_new_can_choose[i] for i in tpark_new_can_choose],
        )  # The selected tpark
        if self.destination_park not in self.accm_tripDpark:
            self.accm_tripDpark[self.destination_park] = 0
        self.accm_tripDpark[self.destination_park] += 1

    def select_new_Opark(self):
        """ When the current traffic zone is not used as the origin by
        the individual, a new one is selected to replace it based on the 
        individual global distribution of trip destinations,
        which may leads to a discontinuity in the trip record"""
        self.bg_tripOpark_pro = self.trans_num_to_pro(self.bg_tripOpark)
        self.accm_tripOpark_pro = self.trans_num_to_pro(self.accm_tripOpark)
        fpark_new_can_choose = {
            k: (
                max(0, v - self.accm_tripOpark_pro[k])
                if k in self.accm_tripOpark_pro
                else v
            )
            for k, v in self.bg_tripOpark_pro.items()
        }
        if sum(fpark_new_can_choose.values()) != 0:
            fpark_new_can_choose = self.trans_num_to_pro(fpark_new_can_choose)
        else:
            fpark_new_can_choose = self.bg_tripOpark_pro
        fpark_new_can_choose_list = list(fpark_new_can_choose.keys())
        self.new_Fpark = np.random.choice(
            fpark_new_can_choose_list,
            p=[fpark_new_can_choose[i] for i in fpark_new_can_choose_list],
        )  # The selected tpark 


 # Generate trips for a single individual
 
def generate_individual_trips(veh: str, f) -> pd.DataFrame:
    """input: individual's id,
       output: the trip records of the individual"""
    vehicel_instance = Individual_trip_demand(veh, macro_control, time_control,macro_control_holiday,time_control_holiday) 
    veh_written = vehicel_instance.name  
    while vehicel_instance.present_time < datetime(2019, 8, 18, 23, 59, 0):
        if vehicel_instance.trip_num_accm > 0:  
            vehicel_instance.present_park = vehicel_instance.destination_park
        doneornot = vehicel_instance.cal_time_slot_can_choose()  
        if doneornot == 'done':
            return 'done'
        if vehicel_instance.ss==1:
            fpark_written = vehicel_instance.present_park
            tpark_written = vehicel_instance.destination_park
            vehicel_instance.cal_specific_time()
            ftime_written = vehicel_instance.present_time

            # write generated trips
            f.write(f"{veh_written},{ftime_written},{fpark_written},{tpark_written}\n")

            vehicel_instance.present_time += timedelta(
                minutes=random.randint(15, 30)
            )


### generating trips 
def generate_trips():
    f = open("./data/scitific_data_glc.csv", "w",encoding='utf-8')
    f.write("hphm,ftime,fpark,tpark\n")
    for v in name_list:
        generate_individual_trips(v, f=f)
        macro_control.cal_time_slot_perfer_pro()
        macro_control_holiday.cal_time_slot_perfer_pro()
        if sum(time_control.accm_time_point_done.values()) % 100 == 0:
            print('weekday',sum(time_control.accm_time_point_done.values()),macro_control.accum_trip_dict)
        if sum(time_control_holiday.accm_time_point_done.values()) % 100 == 0:
            print('holiday:',sum(time_control_holiday.accm_time_point_done.values()), macro_control_holiday.accum_trip_dict)
        if sum(time_control.accm_time_point_done.values()) > trip_num_max and sum(time_control.accm_time_point_done.values()) > trip_num_max_holiday:
            print("Exit through the set upper limit")
            break
    f.close()


def full_path(need_full_df, template_df):
    """Select path according to the origin-destinations of trip"""
    for index, row in need_full_df.iterrows():
        ftpark = row.fpark + "-" + row.tpark
        sel_df = template_df.query(f"ftpark == '{ftpark}'")[["path","traveltime"]]
        if len(sel_df) != 0:
            choose_index = np.random.randint(len(sel_df))
            choose_path = sel_df.iloc[choose_index]['path']
            choose_traveltime = sel_df.iloc[choose_index]['traveltime']
            need_full_df.iloc[index, 4] = choose_path
            need_full_df.iloc[index, 5] = choose_traveltime
    return need_full_df


def generate_path():

    his_path = pd.read_csv("./data/zone_path_TT.csv")[
        ["hphm_new", "fpark", "tpark", "path","traveltime"]
    ]
    his_path = his_path.query(f"hphm_new in {name_list}")
    his_path.insert(3, "ftpark", his_path.fpark + "-" + his_path.tpark)
    generate_data = pd.read_csv("./data/scitific_data_gp.csv", encoding="utf-8") # 
    generate_data.insert(4, "path", 0)
    generate_data.insert(5,"traveltime",0)
    generate_data = generate_data.reset_index(drop=True)
    generate_data_fulled = full_path(generate_data, his_path)
    generate_data_fulled.to_csv("./data/generate_gp_path_tt.csv", encoding='utf-8', index=False)  


if __name__ == "__main__":
    # Determine the travelers to be generated
    generate_label = "High_freq_traveler"  # Select the type of traveler to be generated here
    id_file = open(f"./data/{generate_label}.json", encoding="utf-8")
    veh_mingdan = json.load(id_file)
    name_list = list(veh_mingdan.keys()) 
    get_time_baseline_wANDh("weekday")
    get_time_baseline_wANDh("holiday") 
    macro_control = Macro_control(bg_distribution,fluctate_factor=0.05) # weekday
    macro_control_holiday = Macro_control(bg_distribution_holiday,fluctate_factor=0.05) # holiday
    time_control = Time_ponit_control(1, trip_time_baseline) 
    time_control_holiday = Time_ponit_control(1, trip_time_baseline_holiday)   
    print('weekday:',sum(time_control.time_point_num_baseline.values()))
    print('holiday:', sum(time_control_holiday.time_point_num_baseline.values()))
    generate_trips() # generate 
    generate_path()  # generate trip path


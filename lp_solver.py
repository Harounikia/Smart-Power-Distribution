import pulp
import pandas as pd

class EconomicDispatchSolver:
    """
    ماژول حل مسئله توزیع اقتصادی بار (Economic Dispatch)
    بین نیروگاه‌های حرارتی، خورشیدی و ذخیره‌ساز باتری با استفاده از PuLP
    """
    def __init__(self, demand_mw: float, solar_avail_mw: float, battery_soc_mwh: float):
        self.demand = demand_mw
        self.solar_avail = solar_avail_mw
        self.battery_soc = battery_soc_mwh
        self.model = pulp.LpProblem("Economic_Dispatch", pulp.LpMinimize)
        
    def solve(self, thermal_units: list):
        """
        thermal_units: لیستی از دیکشنری‌های مشخصات نیروگاه‌ها
        [{"name": "G1", "min": 10, "max": 100, "cost_per_mw": 50}, ...]
        """
        # متغیرهای تصمیم
        p_thermal = {
            unit['name']: pulp.LpVariable(f"P_{unit['name']}", lowBound=unit['min'], upBound=unit['max'])
            for unit in thermal_units
        }
        p_solar = pulp.LpVariable("P_Solar", lowBound=0, upBound=self.solar_avail)
        p_bat_dis = pulp.LpVariable("P_Bat_Discharge", lowBound=0, upBound=min(50.0, self.battery_soc))
        
        # هزینه خورشیدی نزدیک به صفر و هزینه استهلاک باتری مشخص
        cost_solar = 5.0  
        cost_battery = 30.0  
        
        # تابع هدف: کمینه‌سازی هزینه کل تولید
        total_cost = (
            pulp.lpSum([p_thermal[u['name']] * u['cost_per_mw'] for u in thermal_units]) +
            p_solar * cost_solar +
            p_bat_dis * cost_battery
        )
        self.model += total_cost, "Total_Operating_Cost"
        
        # قید تعادل توان (مجموع تولید برابر تقاضا)
        demand_constraint = (
            pulp.lpSum([p_thermal[u['name']] for u in thermal_units]) + p_solar + p_bat_dis == self.demand
        )
        self.model += demand_constraint, "Power_Balance_Constraint"
        
        # حل مسئله
        status = self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # استخراج قیمت سایه‌ای (Shadow Price / Dual Value)
        shadow_price = demand_constraint.pi if demand_constraint.pi is not None else 0.0
        
        results = {
            "status": pulp.LpStatus[status],
            "total_cost": pulp.value(self.model.objective),
            "shadow_price": shadow_price,
            "dispatch": {u['name']: p_thermal[u['name']].varValue for u in thermal_units},
            "solar_generation": p_solar.varValue,
            "battery_discharge": p_bat_dis.varValue
        }
        
        return results
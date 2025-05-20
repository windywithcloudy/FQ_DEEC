import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 输入变量：温度（范围 0-40℃）和湿度（范围 0-100%）
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# 输出变量：风速（范围 0-100%）
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# 温度的模糊集合：冷、适中、热
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['moderate'] = fuzz.gaussmf(temperature.universe, 25, 5)
temperature['hot'] = fuzz.smf(temperature.universe, 20, 40)

# 湿度的模糊集合：干燥、中等、潮湿
humidity['dry'] = fuzz.trapmf(humidity.universe, [0, 0, 30, 50])
humidity['medium'] = fuzz.gbellmf(humidity.universe, 10, 3, 50)
humidity['wet'] = fuzz.zmf(humidity.universe, 50, 80)

# 风速的模糊集合：低、中、高
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

temperature.view()
humidity.view()
fan_speed.view()
plt.show()
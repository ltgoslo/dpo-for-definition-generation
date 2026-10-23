import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--r", default=0.5, type=float)
args = parser.parse_args()

z = 1/2*math.log((1.0+args.r)/(1-args.r))
se = math.sqrt(1/34)
lower = z - 1.96 * se
upper = z + 1.96 * se
r_lower = (math.exp(2*lower)-1)/(math.exp(2*lower)+1)
r_upper = (math.exp(2*upper)-1)/(math.exp(2*upper)+1)

print(round(r_lower, 2))  # 0.21 for r = 0.5
print(round(r_upper, 2))  # 0.71 for r = 0.5

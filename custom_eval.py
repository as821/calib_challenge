import numpy as np
import sys


if len(sys.argv) > 1:
  GT_FILE = sys.argv[1]
  INPUT = sys.argv[2]
else:
  raise RuntimeError('No test directory provided')

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))


zero_mses = []
mses = []

gt = np.loadtxt(GT_FILE)
zero_mses.append(get_mse(gt, np.zeros_like(gt)))

test = np.loadtxt(INPUT)
mses.append(get_mse(gt, test))

percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')

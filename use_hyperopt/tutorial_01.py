# 공식 홈페이지 튜토리얼 해보기 
# http://hyperopt.github.io/hyperopt/#getting-started
# 공식 홈페이지 안내

# Getting started
# 다운 완류

import hyperopt

# 목표 함수 정의
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# 탐색 범위 정의
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# objective 최소화하기
from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals = 100)

print(best)
# {'a': 1, 'c2': -0.010374894345358807}
print(hyperopt.space_eval(space, best))
# ('case 2', -0.010374894345358807)
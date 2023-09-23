import torch
import numpy as np
import matplotlib.pyplot as plt
import random
def update_dist(r, support, probs, lim=(-10,10), gamma = 0.8):
    nsup = probs.shape[0]
    vmin, vmax = lim[0],lim[1]
    dz = (vmax-vmin)/(nsup-1.)
    bj = np.round((r-vmin)/dz)
    bj = int(np.clip(bj,0, nsup-1))
    m = probs.copy()
    j = 1
    for i in range(bj,1,-1):
        m[i] += np.power(gamma,j) * m[i-1]
        j += 1
    j = 1
    for i in range(bj, nsup-1,1):
        m[i] += np.power(gamma,j) * m[i-1]
        j += 1
    m /= m.sum()
    return m


def main():
    vmin, vmax = -10,10
    nsup = 51
    support = np.linspace(vmin, vmax, nsup)
    probs = np.ones(nsup)
    probs /= probs.sum()
    z3 = torch.from_numpy(probs).float()
    
    random.seed(42)
    rs = [random.randint(vmin,vmax) for _ in range(10)]
    print(rs)

    plt.subplot(3,4,1)
    plt.tick_params(left = False, right = False , labelleft = False,labelbottom = False, bottom = False)
    plt.bar(support, probs)
    for i in range(10):
        probs = update_dist(rs[i], support, probs, (vmin,vmax),gamma = 0.8)
        plt.subplot(3,4,i+2)
        plt.tick_params(left = False, right = False , labelleft = False,labelbottom = False, bottom = False)
        plt.bar(support,probs)
    plt.show()
        

if __name__ =="__main__":
    main()
		

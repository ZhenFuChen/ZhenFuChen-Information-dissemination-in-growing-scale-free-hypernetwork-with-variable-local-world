#coding=gbk
import random
import numpy as np
import hypernetx as hnx
from scipy import integrate
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

class InfoSpreading(object):
    def funRun(self):
        self.m1 = 3
        self.m2 = 3
        self.m = 1
        self.gamma = 0.1
        self.alpha = 0.4
        self.timestep = 332
        self.yvalue = []

        for beta in range(1,21):
            self.beta = beta/100

            # 使用fsolve函数求解方程
            xita = fsolve(self.equation, 0.5, xtol=1e-6)[0]

            rho = (self.m * (self.m1 + self.m2) * (self.m1 + self.m2 - 1) * self.beta * xita * (1 - xita)) * (self.timestep ** (self.alpha * (self.m1+self.m2)/(self.m2-self.m1 * self.alpha))) / ((1 + self.alpha) * self.gamma * self.m1)

            # print("近似解θ:", xita)
            # 归一化映射
            if self.alpha < 0.1:
                rho_logistic = rho
            elif self.alpha <= 0.2:
                C = 1.0
                rho_logistic = rho / (C + rho) #带尺度的logistic(Scaled Logistic)
            else:
                C = 50.0
                rho_logistic = rho / (C + rho) #带尺度的logistic(Scaled Logistic)
            self.yvalue.append(rho_logistic)
            print("密度ρ:", rho_logistic)
        print(self.yvalue)

    # 定义被积函数
    def integrand(self, k, x):
        return (k ** ((1+self.alpha) * self.m1 / (self.m1 * self.alpha - self.m2))) * (1 / (self.gamma + self.beta * (self.m1 + self.m2 - 1) * k * x))

    # 定义方程
    def equation(self, x):
        result, error = integrate.quad(self.integrand, self.m, float('inf'), args=(x,))
        return result - (self.m2 - self.m1 * self.alpha) * (self.m ** ((1+self.alpha)*self.m1 / (self.m1 * self.alpha - self.m2))) / ((1 + self.alpha) * self.m1 * self.beta * (self.m1 + self.m2 - 1))

    def draw(self):
        t = [i / 100 for i in range(0, 21)]
        t1 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
         0.19, 0.20]

        # yvalue:0,1,2为理论值,3,4,5为仿真结果
        yvalue0 = [0.0,0.21504419553168339, 0.40655963906173993, 0.5458459876151803, 0.643784158584824, 0.7192220895270414, 0.7658521975813403, 0.7928523790128923, 0.8235659410352204, 0.83599538205098124, 0.8432401310440331, 0.8542277826565696, 0.86034649872454696, 0.8613386903289144, 0.8681297386323633, 0.8740468758018641, 0.8792484661430284, 0.8808568169319639, 0.8839678810006952, 0.888657988223452, 0.889886151779038]
        yvalue1 = [0.0,0.6269251070868057, 0.764120984627, 0.8433533177318477, 0.8737215776951418, 0.89781066652249, 0.9085037871328579, 0.9199589896674285, 0.9292763796488515, 0.9379375810798766, 0.9391863452948537, 0.9401591270600685, 0.940938220825419, 0.9415761863565077, 0.9421081571274528, 0.9425585055124999, 0.9429446676762665, 0.9432794460389635, 0.9435724520407311, 0.9438310416355591, 0.94406094001192310]
        yvalue2 = [0.0,0.90014625716,0.9378651776646139, 0.9604842616949554, 0.9670459268717545, 0.9701454895067432, 0.9719470808483102, 0.9731238318733241, 0.9739524253997639, 0.9745673227156143, 0.9750416878181064, 0.9754187259157452, 0.9757255867488106, 0.9759801855832012, 0.9761948202758547, 0.9763782122248656, 0.9765367180029989, 0.9766750797959952, 0.976796907192141, 0.9769049961861137, 0.9770015461333376]

        yvalue3 = [1.3000000000000004e-05, 0.121951, 0.340727, 0.460626, 0.566016, 0.663484, 0.707085, 0.736562, 0.752305, 0.7855739999999998, 0.8006629999999999, 0.813823, 0.824168, 0.8333789999999998, 0.8403589999999999, 0.8465590000000001, 0.852835, 0.8579299999999999, 0.8631230000000001, 0.8669610000000002, 0.8710789999999999]
        yvalue4 = [0.0, 0.586266, 0.69019, 0.8186259999999999, 0.8553459999999999, 0.8764859999999998, 0.8812590000000001, 0.899183, 0.9050060000000001, 0.9184230000000001, 0.9145339999999999, 0.9170640000000002, 0.9282200000000001, 0.929813, 0.92207, 0.9229780000000001, 0.9238370000000002, 0.9251960000000002, 0.9258099999999999, 0.9261030000000001, 0.926824]
        yvalue5 = [2.000000000000001e-05, 0.840675, 0.897711, 0.914576, 0.9317609999999999, 0.9391980000000001, 0.946756, 0.949450000000001, 0.9580699999999999, 0.958266, 0.9589809999999999, 0.9584619999999999, 0.9590789999999997, 0.95921, 0.9594569999999998, 0.9594709999999999, 0.9594040000000001, 0.9589780000000001, 0.958874, 0.958719, 0.958953]

        xlable = [0, 0.1, 0.2, 0.3, 0.4]
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.xticks(xlable)
        # plt.xlabel("β", fontsize=15)
        # plt.ylabel("ρ", fontsize=15)

        plt.scatter(t1, yvalue3, marker="o", s=40, color='#1f77b4')
        plt.scatter(t1, yvalue4, marker="s", s=40, color='#ff7f0e')
        plt.scatter(t1, yvalue5, marker="X", s=40, color='#0ca022')
        plt.plot(t, yvalue0, "-", color='#1f77b4')
        plt.plot(t, yvalue1, "-", color='#ff7f0e')
        plt.plot(t, yvalue2, "-", color='#0ca022')

        ax.legend(labels=[r"$α=0$","$α=0.2$","$α=0.4$","theoretical($α=0$)","theoretical($α=0.2$)","theoretical($α=0.4$)"], ncol=1, fontsize=20)

        plt.xlabel("β", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        #plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

if __name__ == '__main__':
    infospr = InfoSpreading()
    #infospr.funRun()
    infospr.draw()

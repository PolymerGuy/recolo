import numpy as np
import matplotlib.pyplot as plt

n_pts = 20
xsi1,xsi2 = np.meshgrid(np.linspace(-1,1,n_pts),np.linspace(-1,1,n_pts))

# shape function
Np1 = 1 / 4 * (1 - xsi1) ** 2 * (2 + xsi1)
Nq1 = 1 / 4 * (1 - xsi2) ** 2 * (2 + xsi2)
Np3 = 1 / 4 * (1 + xsi1) ** 2 * (2 - xsi1)
Nq3 = 1 / 4 * (1 + xsi2) ** 2 * (2 - xsi2)
# 1st derivative of shape function
Bp1 = 1 / 4 * (-3) * (1 - xsi1 ** 2)
Bq1 = 1 / 4 * (-3) * (1 - xsi2 ** 2)
Bp3 = 1 / 4 * (+3) * (1 - xsi1 ** 2)
Bq3 = 1 / 4 * (+3) * (1 - xsi2 ** 2)
# 2nd derivative of shape function
Cp1 = 1 / 4 * (6 * xsi1)
Cq1 = 1 / 4 * (6 * xsi2)
Cp3 = 1 / 4 * (-6 * xsi1)
Cq3 = 1 / 4 * (-6 * xsi2)
# virtual displacement
# w = np.ones((4, iprw**2))
w = np.array([Np1 * Nq1, Np3 * Nq1, Np3 * Nq3, Np1 * Nq3])

plt.imshow(w[3,:,:])
plt.show()
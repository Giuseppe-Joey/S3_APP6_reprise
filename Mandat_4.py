import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.solvers import solve

def matrice_multiplyer(M1, M2):

    OO = M1[0][0]*M2[0][0] + M1[0][1]*M2[1][0]
    OI = M1[0][0]*M2[0][1] + M1[0][1]*M2[1][1]
    IO = M1[1][0]*M2[0][0] + M1[1][1]*M2[1][0]
    II = M1[1][0]*M2[0][1] + M1[1][1]*M2[1][1]

    M = [[OO, OI], [IO, II]]
    #print(M)
    return M


def mandat_4():
    Cap = 5.193894936952667e-06
    L = 0.3477490059621899
    Res = 5
    Vll = 500000
    Eg = 13800
    P_max = ((6 * 350000000) - 100000000) * 0.9
    Freq = 60
    SIL = 956471986.1492959
    a = 13800 / 500000

    Znom = ((Vll) ** 2) / (6 * (350e6))
    # print("Znom = ", Znom)

    pui = 0
    Vb3 = 0
    Ib3 = 0
    Vb2 = 0
    Angle = 0
    Vari = 0
    P_vari = 20000000
    Pourcentage = 0


    for i in range(100):
        i = i/100
        a = 13800 / 500000



        Ma = [[1, 0], [0, 1]]

        pu = Znom

        RMtY = 500
        XMtY = 500.01j
        MtZ = (RMtY * XMtY) / (RMtY + XMtY)
        MtY = 1 / (MtZ * pu)

        MtZ1 = (0.002 + 0.08j) * pu
        MtZ2 = (0.00432 + 0.172j) * pu
        Mt = [[1 + (MtY * MtZ1), MtZ1 + MtZ2 + (MtY * MtZ1 * MtZ2)], [MtY, 1 + (MtY * MtZ2)]]

        MlZ = (L * 2 * np.pi * 60 * 1j) + Res

        MlY1 = 1 / (1 / (Cap * 2 * np.pi * 60 * 1j))

        Ml = [[1 + ((MlZ * MlY1) / 2), MlZ], [MlY1 * (1 + ((MlZ * MlY1) / 4)), 1 + ((MlZ * MlY1) / 2)]]

        Mc = [[1, 0], [-i*MlY1, 1]]



        Y = -i*MlY1

        if Y != 0 :
            Z = 1/Y
            L2 = Z/(2*np.pi*60)
        else:
            L2 = 0

        # Multipliquation des matrices#
        ########################################
        Mat = matrice_multiplyer(Ma, Mt)
        Matl = matrice_multiplyer(Mat, Ml)
        Matlc = matrice_multiplyer(Matl, Mc)
        # print(Matlc)

        Mlc = matrice_multiplyer(Ml, Mc)

        AA = Mlc[0][0]
        BB = Mlc[0][1]

        A = Matlc[0][0]
        B = Matlc[0][1]
        C = Matlc[1][0]
        D = Matlc[1][1]

        Ar = np.real(A)
        Aim = np.imag(A)
        Br = np.real(B)
        Bim = np.imag(B)


        x = symbols('x', real=True)

        Temporaire = (Vll / np.sqrt(3)) ** 2 - ((Ar * x) + (((Br * P_vari) / (3 * x)))) ** 2 - (
        ((Aim * x) + (Bim * P_vari) / (3 * x))) ** 2

        Vb3__ = solve(Temporaire, x)
        # print(Vb3__)

        if Vb3__ != []:
            Vb3_ = Vb3__[3]

            pui_ = P_vari

            Pourcentage = np.append(Pourcentage,i)

            Ib3_ = pui_ / (3 * Vb3_)

            Vb2_ = (AA * Vb3_) + (BB * Ib3_)
            Vb2_ = complex(Vb2_)

            # print(Vb2_)

            Vari_ = np.absolute((np.abs(Vb2_) - np.abs(Vb3_)) / np.absolute(Vb2_))
            Vari = np.append(Vari, Vari_)   
            print("Pourcentage", i, "V.T.", Vari_, "inductance equivalente", L2, "VB2 = ", np.absolute(Vb2_), "Vb3 = ", Vb3_, )



            #Angle_Vb2_ = np.angle(Vb2_)

            #Angle_ = (Angle_Vb2_ * 360) / (2 * np.pi)
            #Angle = np.append(Angle, Angle_)
            # print(Angle)
            #s = False
            #if s:
                #if Vari_ < 0.05 or Vari_ > -0.05:
                    #if Vb3_ < (1.05 * Vll / np.sqrt(3)) or Vb3_ > (0.95 * Vll / np.sqrt(3)):
                        #if Angle_ < 35:
                            #if Ib3_ < 1750:
                                # Bonne_val[i] = np.append(Bonne_val[i],P_vari)
                                #print("No de condensateur = ", i, "    Valeur de puissance", P_vari)


        else:
            break

    plt.plot(Pourcentage[1:], Vari[1:])
    plt.title("V.T. en fonction de puissance")

    plt.show()

mandat_4()

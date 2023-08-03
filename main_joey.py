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






def mandat_1 ():

    print("---------------------------------------------------")
    print("-----------------------MANDAT 1--------------------")
    print("---------------------------------------------------")
    #Specs Systeme
    Nb_moteur = 6
    P_moteur = 350000000
    V_moteur = 13800
    Tension_transportee = 500000
    P_consomé_localement = 100000
    Puissance_transferee  = 0.9
    distance = 400000
    Variation = 0.05

    #Voir systeme pi

    #Specs fil
    Nb_fil_par_faisseau = 4
    Distance_entre_les_fils = 0.45
    Distance_entre_les_faisseau = 12
    GMR0_en_metre = 0.0479/3.28
    print("Le GMR0 en metres est: ", GMR0_en_metre)
    Capacite_transport_courant = 1250
    Resistance_petit_courant_par_mile = 0.0695   # 25 degre
    Resistance_grand_courant_par_mile = 0.0803   # 50 degre
    #inductance_par_miles = 0.369
    #Shunt_capacitance_par_miles = 0.0838

    # Calcul Capacitance par phase
    Deq = np.cbrt(12*12*24)
    print("Le Deq en metres est: ", Deq)

    Dsc = 1.091 * np.sqrt(np.sqrt(GMR0_en_metre * (Distance_entre_les_fils**3)))
    print("Le Dsc en metres est: ", Dsc)

    # epsilon0 = (1/(36*np.pi))*(10^-9)
    # print("Epsilon 0 est: ", epsilon0)

    Can_par_metre = (2 * np.pi * 8.854e-12) / np.log(Deq/Dsc)
    print("Le Can par metre est: ", Can_par_metre)

    Can_tot = distance * Can_par_metre
    print("Voici la capacité ligne neutre Can_tot = ", Can_tot)





    # Calcul Reactance inductive
    GMD = np.cbrt(12 * 12 * 24)
    print("Le GMD en metres est: ", GMD)

    GMR01 = GMR0_en_metre * np.exp(-1/4)
    print("Le GMR0' en metres est:  = ", GMR01)

    GMR1 = np.sqrt(np.sqrt(GMR01 * Distance_entre_les_fils ** 3)) * 1.091       #lerreur etait ici il manquait le *1.091
    print("Le GMR' en metres est:  = ", GMR1)

    La_par_metre = (2e-7) * np.log(GMD/GMR1)
    print("Linductance par metre est = ", La_par_metre)

    La = La_par_metre * distance
    print("Inductance par phase est de La = ", La)





    # Calcul Resistance  a 50 degre
    res_1_fil = (0.0803*400)/(1.603)    #on convertie par kilometre au lieu de par mile
    res_par_phase = res_1_fil/4
    print("La resistance par phase est de ", res_par_phase)


    #effet corona

    Q_div_const = (Can_par_metre * (Tension_transportee / np.sqrt(3))) / (8 * np.pi * 8.854e-12)
    print("La portion constante de la formule de E_rmax est ", Q_div_const)
    Variable = (1/GMR0_en_metre) + (1/(Distance_entre_les_fils*np.sqrt(2))) + (2*np.cos(np.pi/4)/Distance_entre_les_fils)
    E_max = Q_div_const * Variable
    E_max = E_max/100       # pour lavoir en kV par cm
    print("Voltage max a la surface des fils en kV/cm = ", E_max/1000)

    Q = Can_par_metre*Tension_transportee*np.sqrt(3)

    print("---------------------------------------------------")
    print("\n\n\n")






def mandat_2():

    print("---------------------------------------------------")
    print("-----------------------MANDAT 2--------------------")
    print("---------------------------------------------------")

    Cap = 5.193894936952667e-06         # en F/m
    L = 0.3477490059621899              # en H/m
    Res = 12.87209
    Vll = 500000                        # en V

    Zc = np.sqrt(L/Cap)
    print("Impedence caracteristique Zc = ", Zc)

    SIL = Vll**2/Zc
    print("Surge impedence loading SIL = ", SIL)

    Ib3 = SIL/(np.sqrt(3)*Vll)
    print("Courant de ligne Ib3 = ", Ib3)

    Angle_dephasage1 = np.arcsin((SIL*L)/np.square(Vll))
    print("Angle_dephasage Phi 1= ", Angle_dephasage1)

    Angle_dephasage = (360*Angle_dephasage1)/(2*np.pi)
    print("Angle_dephasage Phi = ", Angle_dephasage)

    Vb2 = ((1+(1j*L*Cap))*Vll)+(L*Ib3)
    print("tension a lentree Vs = ",np.real(Vb2))

    VT = (np.real(Vb2)-Vll)/np.real(Vb2)
    print("Variation de tension V.T. = ", VT*100,"%")

    Vb3_pu = Vll/Vb2
    print("Vb3_pu = ",np.real(Vb3_pu))

    print("---------------------------------------------------")
    print("\n\n\n")










def mandat_3():

    print("---------------------------------------------------")
    print("-----------------------MANDAT 3--------------------")
    print("---------------------------------------------------")

    Cap = 5.193894936952667e-06
    L = 0.3477490059621899
    Res = 5
    Vll = 500000
    Eg = 13800
    P_max = ((6*350000000)-100000000)*0.9
    Freq = 60
    # SIL = 966170204.5010189
    a = 13800/500000

    Zc = np.sqrt(L/Cap)
    SIL = Vll**2/Zc
    print("SIL=", SIL)

    Znom = ((Vll)**2)/(6*(350e6))
    #print("Znom = ", Znom)

    Cond_comp = 1/(np.square(2*np.pi*Freq)*L)
    print("Condensateur de compensation equivalent C = ", Cond_comp*1e6, "uF")
    SS = True
    if SS:
        plt.figure()
        aa = plt.subplot(1, 1, 1)
        plt.figure()
        bb = plt.subplot(1, 1, 1)
        plt.figure()
        cc = plt.subplot(1, 1, 1)
        plt.figure()
        dd = plt.subplot(1, 1, 1)
        plt.figure()
        ee = plt.subplot(1, 1, 1)

    for i in range(8):
        a = 13800/500000

        #creation des matrices
     ####################################
        if i==0:
            Xc = 0
        else:
            Xc = (2*np.pi*60*(0.1*i*L))

        Ma = [[1, 0], [0, 1]]


        pu = Znom

        RMtY = 500
        XMtY = 500.01j
        MtZ = ((RMtY*XMtY)/(RMtY+XMtY))*pu
        MtY = 1/(MtZ)


        MtZ1 = (0.002+0.08j)*pu
        MtZ2 = (0.00432+0.172j)*pu
        Mt = [[1+(MtY*MtZ1), MtZ1+MtZ2+(MtY*MtZ1*MtZ2)], [MtY, 1+(MtY*MtZ2)]]


        MlZ = (L*2*np.pi*60*1j)+Res
        MlY1 = 1/(1/(Cap*2*np.pi*60*1j))
        Ml = [[1+((MlZ*MlY1)/2), MlZ], [MlY1*(1+((MlZ*MlY1)/4)), 1+((MlZ*MlY1)/2)]]
        print("Ml = ", Ml)

        Mc = [[1, -1j*Xc], [0, 1]]
        #print(np.real(Xc))

        #print("Mc = ", Mc)

            #Multipliquation des matrices#
        ########################################
        Mat = matrice_multiplyer(Ma, Mt)
        Matl = matrice_multiplyer(Mat, Ml)
        Matlc = matrice_multiplyer(Matl, Mc)
        #print(Matlc)

        Mlc = matrice_multiplyer(Ml, Mc)

        AA = Mlc[0][0]
        BB = Mlc[0][1]


        A =  Matlc[0][0]
        B =  Matlc[0][1]
        C =  Matlc[1][0]
        D =  Matlc[1][1]
        
        Ar = np.real(A)
        Aim = np.imag(A)
        Br = np.real(B)
        Bim = np.imag(B)

        pui = 0
        Vb3 = 0
        Ib3 = 0
        Vb2 = 0
        Angle = 0
        Vari = 0
        b = False
        for P_vari in range(int(0.8*SIL), int(P_max), 10000000):


            x = symbols('x', real=True)

            Temporaire = (Vll/np.sqrt(3))**2-((Ar*x)+(((Br*P_vari)/(3*x))))**2-(((Aim*x)+(Bim*P_vari)/(3*x)))**2

            Vb3__ = solve(Temporaire, x)
            #print(Vb3__)

            if Vb3__ != []:
                Vb3_ = Vb3__[3]
                Vb3 = np.append(Vb3, Vb3_)

                pui_ = P_vari
                pui = np.append(pui, pui_)

                Ib3_ = pui_ / (3 * Vb3_)
                Ib3 = np.append(Ib3, Ib3_)

                Vb2_ = (AA * Vb3_) + (BB * Ib3_)
                Vb2_= complex(Vb2_)
                Vb2 = np.append(Vb2, Vb2_)

                #print(Vb2_)

                Vari_ = np.absolute((np.abs(Vb2_) - np.abs(Vb3_)) / np.absolute(Vb2_))
                Vari = np.append(Vari, Vari_)

                Angle_Vb2_ = np.angle(Vb2_)

                Angle_ = (Angle_Vb2_ * 360) / (2 * np.pi)
                Angle = np.append(Angle, Angle_)


                s = True
                if s:
                    if Vari_<0.05 or Vari_>-0.05:
                        if Vb3_<(1.05*Vll/np.sqrt(3)) or Vb3_>(0.95*Vll/np.sqrt(3)):
                            if Angle_< 35:
                                if Ib3_<1750:
                                    # Bonne_val[i] = np.append(Bonne_val[i],P_vari)
                                    print("No de condensateur = ",i, "    Valeur de puissance",P_vari, "   Vb2 = ", np.absolute(Vb2_), "Vb3 = ", Vb3_, "I = ", Ib3_)


            else:
                break

        aa.plot(pui[1:], Vb3[1:]/(Vll/np.sqrt(3)))
        aa.set_title("Vb3 en fonction de puissance")
        aa.set
        aa.set_ylabel('Vb3')
        aa.legend(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], loc="upper right")

        bb.plot(pui[1:], np.absolute(Ib3[1:]))
        bb.set_title("Ib3 en fonction de puissance")
        bb.legend(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], loc="upper right")

        cc.plot(pui[1:], np.absolute(Vb2[1:]))
        cc.set_title("Vb2 en fonction de puissance")
        cc.legend(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], loc="upper right")

        dd.plot(pui[1:], Vari[1:])
        dd.set_title("V.T. en fonction de puissance")
        dd.legend(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], loc="upper right")

        ee.plot(pui[1:], Angle[1:])
        ee.set_title("Angle en fonction de puissance")
        ee.legend(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%'], loc="upper right")


    plt.show()

    print("---------------------------------------------------")
    print("\n\n\n")














def mandat_4():

    print("---------------------------------------------------")
    print("-----------------------MANDAT 4--------------------")
    print("---------------------------------------------------")


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




mandat_1()
mandat_2()
mandat_3()
mandat_4()

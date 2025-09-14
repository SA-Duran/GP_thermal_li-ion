from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pybamm

def _initial_concentrations_for_mode(param: pybamm.ParameterValues, mode: str) -> None:
    if mode == "Charge":
        param["Initial concentration in negative electrode [mol.m-3]"] = 4800
        param["Initial concentration in positive electrode [mol.m-3]"] = 48000
    else:
        param["Initial concentration in negative electrode [mol.m-3]"] = 22400
        param["Initial concentration in positive electrode [mol.m-3]"] = 27300

def _solve_once(param: pybamm.ParameterValues, experiment, durations_s):
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"}, name="lumped thermal model")
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol = sim.solve(durations_s)
        return sol
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None

def _rho_eff(param: pybamm.ParameterValues) -> float:
    components = ["Negative current collector", "Negative electrode", 
                  "Separator", "Positive electrode", "Positive current collector"]
    rho_k = np.array([param.get(f"{comp} density [kg.m-3]") for comp in components])
    cp_k  = np.array([param.get(f"{comp} specific heat capacity [J.kg-1.K-1]") for comp in components])
    L_k   = np.array([param.get(f"{comp} thickness [m]") for comp in components])
    return float(np.sum(rho_k * cp_k * L_k) / np.sum(L_k))

def run_experiment_grid(cfg: Dict[str, Any]) -> List[list]:
    temps = cfg["grid"]["temperatures_C"]
    modes = cfg["grid"]["modes"]
    c_rates = cfg["grid"]["c_rates"]
    dur = cfg["experiment"]["durations_s"]
    stop_v_charge = cfg["experiment"]["stop_voltage_charge"]
    stop_v_discharge = cfg["experiment"]["stop_voltage_discharge"]

    rows: List[list] = []
    for tempC in temps:
        for c_rate in c_rates:
            for mode in modes:
                param = pybamm.ParameterValues("Marquis2019")
                param["Initial temperature [K]"] = tempC + 273.15
                param["Ambient temperature [K]"] = tempC + 273.15
                h = param.get("Total heat transfer coefficient [W.m-2.K-1]")

                stop_v = stop_v_charge if mode == "Charge" else stop_v_discharge
                step_str = f"{mode} at {c_rate}C until {stop_v} V"
                experiment = pybamm.Experiment([step_str])

                _initial_concentrations_for_mode(param, mode)

                sol = _solve_once(param, experiment, dur)
                if sol is None:
                    continue

                # Extract variables
                J_n = sol["Negative electrode interfacial current density [A.m-2]"].entries.mean(axis=0)
                J_p = sol["Positive electrode interfacial current density [A.m-2]"].entries.mean(axis=0)
                V = sol["Terminal voltage [V]"].entries
                t = sol["Time [s]"].entries
                I = sol["Current [A]"].entries
                soc = (V - 3.105) / 0.995
                T = sol["Volume-averaged cell temperature [K]"].entries
                P = sol["Terminal power [W]"].entries
                q_ohmic = sol["Volume-averaged Ohmic heating [W.m-3]"].entries
                q_rev = sol["Volume-averaged reversible heating [W.m-3]"].entries
                q_irr = sol["Volume-averaged irreversible electrochemical heating [W.m-3]"].entries
                q_contact = sol["Lumped contact resistance heating [W.m-3]"].entries
                q_total = sol["Volume-averaged total heating [W.m-3]"].entries
                q_loss = sol["Surface total cooling [W.m-3]"].entries
                q_ratio = q_rev / q_irr
                C_rate = sol["C-rate"].entries
                T_amb = sol["Ambient temperature [K]"].entries 
                h_T_amb = T_amb * h / 10000

                rho_eff = _rho_eff(param)
                dT_dt_mod = (q_total + q_loss)/rho_eff
                dT_dt_est = np.gradient(T, t)

                V_ocv_model = sol["Battery open-circuit voltage [V]"].entries
                R= sol["Resistance [Ohm]"].entries
                R_n = sol["X-averaged negative electrode resistance [Ohm.m2]"].entries
                R_p = sol["X-averaged positive electrode resistance [Ohm.m2]"].entries
                n_ocv_n = sol["Negative electrode bulk open-circuit potential [V]"].entries
                n_ocv_p = sol["Positive electrode bulk open-circuit potential [V]"].entries
                V_ocv = n_ocv_p - n_ocv_n
                V_ocv_dt= np.gradient(V_ocv, t)

                n_conc_n= sol["Negative particle concentration overpotential [V]"].entries
                n_conc_p= sol["Positive particle concentration overpotential [V]"].entries
                V_conc_p= n_conc_p - n_conc_n
                V_conc_p_dt= np.gradient(V_conc_p, t)
                dif_conc_p = n_conc_p + n_conc_n
                dif_conc_p_dt = np.gradient(dif_conc_p, t)

                n_rxn_n = sol["X-averaged negative electrode reaction overpotential [V]"].entries
                n_rxn_p = sol["X-averaged positive electrode reaction overpotential [V]"].entries
                V_rxn = n_rxn_p - n_rxn_n
                V_rxn_dt= np.gradient(V_rxn,t)
                dif_rxn = n_rxn_p + n_rxn_n
                dif_rxn_dt= np.gradient(V_rxn,t)

                i_o_n= sol["X-averaged negative electrode exchange current density [A.m-2]"].entries
                i_o_n_dt= np.gradient(i_o_n,t)
                i_o_p= sol["X-averaged positive electrode exchange current density [A.m-2]"].entries
                i_o_p_dt =np.gradient(i_o_p,t)

                V_conc_ele= sol["X-averaged concentration overpotential [V]"].entries
                V_conc_ele_dt = np.gradient(V_conc_ele,t)

                V_ohm_ele= sol["X-averaged electrolyte ohmic losses [V]"].entries
                V_ohm_ele_dt = np.gradient(V_ohm_ele,t)

                n_ohm_n= sol["X-averaged negative electrode ohmic losses [V]"].entries
                n_ohm_p= sol["X-averaged positive electrode ohmic losses [V]"].entries
                V_ohm_p= n_ohm_p-n_ohm_n
                V_ohm_p_dt= np.gradient(V_ohm_p,t)
                dif_ohm_p= n_ohm_p+n_ohm_n
                dif_ohm_p_dt = np.gradient(dif_ohm_p,t)

                estado = np.ones(len(t)) if mode == "Charge" else np.zeros(len(t))
                soc_dis=sol["Discharge capacity [A.h]"].entries

                V_conc_total=V_conc_p+V_conc_ele
                n_rxn_conc= (V_rxn/(V_conc_total))/100
                IT = I*T
                T_norm= T/273.15
                IT_ovp= T_norm*I*(V_conc_ele_dt+V_conc_p_dt+V_ohm_ele_dt+V_rxn_dt)

                dQ_dV = np.gradient(soc_dis, V)
                dV_dt= np.gradient(V,t)

                dT_Qohm_dt = q_ohmic/rho_eff
                dT_Qirr_dt = q_irr/rho_eff
                dT_Qloss_dt = q_loss/rho_eff
                dT_Qloss_ohm_irr_dt= dT_Qohm_dt+dT_Qirr_dt+dT_Qloss_dt
                dT_dt_mod_loss = dT_dt_mod - dT_Qloss_dt

                for i in range(len(t)):
                    rows.append([
                        V[i], t[i], I[i], soc[i], T[i], P[i], soc_dis[i],
                        q_ohmic[i], q_rev[i], q_irr[i],
                        q_contact[i], q_loss[i], q_ratio[i],
                        C_rate[i], T_amb[i], estado[i], q_total[i],
                        dT_dt_est[i], dT_dt_mod[i], tempC + 273.15,
                        V_ocv_model[i], V_ocv[i], V_ocv_dt[i],
                        V_conc_p[i], V_conc_p_dt[i], V_rxn[i],
                        V_rxn_dt[i], V_conc_ele[i] , V_conc_ele_dt[i],
                        V_ohm_ele[i], V_ohm_ele_dt[i], V_ohm_p[i], V_ohm_p_dt[i],
                        n_rxn_conc[i],
                        dif_conc_p[i]  ,dif_conc_p_dt[i],
                        dif_rxn[i], dif_rxn_dt[i],
                        dif_ohm_p[i], dif_ohm_p_dt[i],
                        IT[i], IT_ovp[i], T_norm[i],
                        dQ_dV[i], dV_dt[i],
                        dT_Qohm_dt[i], dT_Qirr_dt[i], dT_Qloss_dt[i], dT_Qloss_ohm_irr_dt[i],
                        dT_dt_mod_loss[i]
                    ])
    return rows

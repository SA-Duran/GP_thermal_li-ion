from __future__ import annotations

# Etiquetas cortas (mathtext) para columnas del dataset
FEATURE_LABELS = {
    "Terminal voltage time derivative [V/s]": r"$\dot{V}$",
    "Reaction overpotential time derivative [V/s]": r"$\dot{\eta}_{\mathrm{rxn}}$",
    "Electrode concentration overpotential time derivative [V/s]": r"$\dot{\eta}_{s\text{-}cnc}$",
    "Electrolyte concentration overpotential time derivative [V/s]": r"$\dot{\eta}_{\ell\text{-}cnc}$",
    "Electrolyte ohmic overpotential time derivative [V/s]": r"$\dot{\eta}_{\ell\text{-}ohm}$",
}

# Etiqueta corta para el target
TARGET_LABEL = r"$\dot{T}$"

# Mapeo para nombres de hiperparámetros (Sobol-HP)
# Regla: "misma variable" + guion, usando \ell para lengthscale y \sigma para varianzas
HP_LABELS = {
    "kV_lengthscale":        r"$\ell$–$\dot{V}$",
    "kV_variance":           r"$\sigma$–$\dot{V}$",
    "k1_lengthscale_1":      r"$\ell$–$\dot{\eta}_{\mathrm{rxn}}$",
    "k1_lengthscale_2":      r"$\ell$–$\dot{\eta}_{s\text{-}cnc}$",
    "k1_lengthscale_3":      r"$\ell$–$\dot{\eta}_{\ell\text{-}cnc}$",
    "k1_lengthscale_4":      r"$\ell$–$\dot{\eta}_{\ell\text{-}ohm}$",
    # Varianza del bloque de sobrepotenciales (k1). Si prefieres otra notación, cámbiala aquí.
    "k1_variance":           r"$\sigma$–$\dot{\eta}_{\mathrm{ovp}}$",
    "noise_variance":        r"$\sigma$–noise",
}

def prettify(name: str) -> str:
    """Devuelve la etiqueta bonita si existe, o el nombre original si no."""
    return FEATURE_LABELS.get(name, HP_LABELS.get(name, name))

def prettify_list(names: list[str]) -> list[str]:
    return [prettify(n) for n in names]

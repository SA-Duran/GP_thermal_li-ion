from __future__ import annotations
import gc, time

def hard_cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    time.sleep(1)  # pequeña pausa ayuda en Windows a liberar handles

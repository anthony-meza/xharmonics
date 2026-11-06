# Notes

- `fit` returns coefficients only.
- `evaluate` reconstructs the harmonic contributions from those coefficients.
- `harmonic=0` is the mean by construction.
- `basis` indexes the real Fourier basis functions `"cos"` and `"sin"`.
- Datetime-like coordinates are converted to fractional years internally.
- daily and monthly datetime-like data are supported.
- Numeric time requires an explicit `fundamental_period` during fitting. This is
  the fundamental period, i.e. the longest fitted period.

## Calendar and leap-year behavior

- Datetime-like coordinates are converted using elapsed days expressed in fractional years.
- Leap years therefore affect the exact phase position of daily and monthly timestamps through elapsed time.
- For `cftime`, the conversion uses calendar-specific days per year for `360_day`, `noleap`/`365_day`, and `all_leap`/`366_day`, and otherwise falls back to `365.2425`.
- Coefficient datasets retain the fitted calendar, and `evaluate` warns on calendar mismatches.
- Cross-calendar evaluation should still be treated as provisional behavior in this version.

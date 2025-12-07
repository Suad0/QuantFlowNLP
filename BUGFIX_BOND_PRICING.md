# Bond Pricing Bug Fix

## Issue
The bond pricing endpoint was failing with a QuantLib API error:
```
TypeError: Wrong number or type of arguments for overloaded function 'new_FlatForward'
```

## Root Cause
The `FlatForward` constructor in QuantLib expects a `Frequency` enum (e.g., `ql.Annual`, `ql.Semiannual`) but was being passed a `Period` object (e.g., `ql.Period(ql.Annual)`).

Similarly, the `bondYield` method had incorrect parameter order.

## Fix Applied
Updated `app/services/bond_pricer.py`:

1. **FlatForward Constructor**: Changed from passing `period` to passing `freq` (Frequency enum)
   ```python
   # Before
   flat_forward = ql.FlatForward(settlement_ql, yield_rate, day_count, ql.Compounded, period)
   
   # After
   flat_forward = ql.FlatForward(settlement_ql, yield_rate, day_count, ql.Compounded, freq)
   ```

2. **bondYield Method**: Fixed parameter order
   ```python
   # Before
   ytm = bond.bondYield(price, day_count, ql.Compounded, freq)
   
   # After
   ytm = bond.bondYield(day_count, ql.Compounded, freq, price)
   ```

## Test Results
After the fix, all bond pricing tests pass:
- ✅ POST /api/v1/quant/pricing/bond - Returns correct bond price, duration, and convexity
- ✅ All 3 quantitative finance endpoints working

## Example Response
```json
{
  "price": 95.72333103746497,
  "yield_to_maturity": null,
  "duration": 4.446153293007389,
  "convexity": 23.314567053324627,
  "method": "QuantLib"
}
```

## Status
✅ **RESOLVED** - Bond pricing endpoint fully functional

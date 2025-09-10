# Julia’s Idiosyncrasies

## Anecdotes 
- Packages aren’t usually maintained. But most logic can be tersely rewritten in ideomatic Julia.

## Mixing named and unnamed parameters significantly degrades performance. 
### All-keyword call can be fully optimized
When keyword arguments are the *only* parameters and the call site supplies a compile-time-known NamedTuple of keywords, the compiler specializes on that NamedTuple type, inlines the kw-wrapper, and scalar-replaces the NamedTuple (no heap allocation). Hence f_named can match positional speed.
### Mixed (positional+keyword) goes through the kw-wrapper path
Calls of the form f(x,y; z=…) are lowered to a hidden Core.kwfunc(f) plus a keyword-sorter/merger (kwsorter) that builds a NamedTuple and passes it to the wrapper, which then forwards to the actual method. In current Julia, that path is not fully eliminated for mixed calls.
```julia
using BenchmarkTools

f_pos(x, y, z) = x * y + z
f_named(; x, y, z) = x * y + z
f_mixed(x, y; z) = x * y + z

x, y, z = 3.0, 4.0, 5.0
@btime f_pos($x, $y, $z) #  2.093 ns (0 allocations: 0 bytes)
@btime f_named(x=$x, y=$y, z=$z) #  2.154 ns (0 allocations: 0 bytes)
@btime f_mixed($x, $y, z=$z) #   29.694 ns (4 allocations: 64 bytes)
```


## Intermediate functions vs variables have no performance differences

Julia’s compiler:
1. **Inlining** – If a small helper function (like g_fn) is called from another function, Julia will inline its body at compile time, eliminating the function call completely.
2. **Constant Propagation + SROA (Scalar Replacement of Aggregates)** – Temporaries, intermediate results, and small structs are optimized away, so the generated machine code just has raw arithmetic operations.
```julia
using BenchmarkTools

# --- Intermediate function style (inlined) ---
f_fn(x, y, z) = g_fn(x, y) + z
g_fn(x, y) = x * y

# --- Intermediate function style (noinline) ---
f_fn_noinline(x, y, z) = g_fn_noinline(x, y) + z
@noinline g_fn_noinline(x, y) = x * y

# --- Intermediate variable style ---
function f_var(x, y, z)
    tmp = x * y
    return tmp + z
end

x, y, z = 3.0, 4.0, 5.0

@btime f_fn($x, $y, $z) #  2.113 ns (0 allocations: 0 bytes)
@btime f_var($x, $y, $z) #  2.154 ns (0 allocations: 0 bytes)
@btime f_fn_noinline($x, $y, $z) #  3.336 ns (0 allocations: 0 bytes)
```


---
title: Julia's Idiosyncrasies
layout: default
permalink: /julias-idiosyncrasies/
---

# Julia’s Idiosyncrasies

## Anecdotes 
- Packages aren’t usually maintained
- Julia version 1.10 is after Julia v1.9
- Packages are case sensitive. Unlike pip or gem, `add cuda` != `add CUDA`

## Syntax
### The same function can mean and do different things in different contexts
It’s ideomatic to write functions such that`!` is either ‘not’ or ‘mutate’.
```julia
!true #=> false
something = 3
change_it(something) #=> outputs something
something == 3 #=> true
change_it!(something) #=> when a function has ! in it's name, it signals that it'll mutate this variable.
something == 3 #=> false
```

`.`  can refer to field access or broadcasting depending on the context. 
```julia
struct Layer 
           W::Matrix
           b::Vector
           activation::Function
       end

(d::Layer)(x) = d.activation.(d.W*x .+ d.b)
```

`d.activation`, `d.W`, `d.b` are refering to the fields inside the Layer struct. The other `.`’s are broadcast operations.

Function overloading is ideomatic and it feels similar to that paradigm.

## Single Value Tuples have commas
A metaprogramming gotcha, because `(` and `)` can infer both operations order and Tuple definition depending what’s inside the parenthesis.
```julia
int = 42
parenthesized_value = (((42)))
tuple = (42,)

display(typeof(int)) #Int64
display(typeof(parenthesized_value)) #Int64
display(typeof(tuple)) #Tuple{Int64}

display(typeof(int) == typeof(parenthesized_value)) #true
display(Meta.parse("(42)") == Meta.parse("(42,)")) #false
```

## Performance
### Packages can pollute Global namespaces with `using`
* This is resolved by just using `import` keyword
* Import conflicts are caught by the compiler
* After a year in Julia, I think this is more convenient
```julia
sigmoid #ERROR: UndefVarError: `sigmoid` not defined in `Main`

using Flux #the ideomatic way of importing packages into Julia
sigmoid #σ (generic function with 2 methods)
sigmoid === Flux.sigmoid #true
```

### Mixing named and unnamed parameters significantly degrades performance. 
#### All-keyword call can be fully optimized
When keyword arguments are the *only* parameters and the call site supplies a compile-time-known NamedTuple of keywords, the compiler specializes on that NamedTuple type, inlines the kw-wrapper, and scalar-replaces the NamedTuple (no heap allocation). Hence f_named can match positional speed.
#### Mixed (positional+keyword) goes through the kw-wrapper path
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

### Intermediate functions vs variables have no performance differences

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

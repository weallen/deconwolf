# PSF Sizing Strategies in dwpy

## Overview

dwpy provides **three PSF sizing strategies** to handle different use cases and maintain compatibility with the C implementation.

## The Three Modes

### 1. Physical Mode (Recommended) ⭐

**Maintains constant physical size** regardless of pixel resolution.

```python
xy, z = dwpy.auto_psf_size(dxy, dz, NA, wvl, ni, mode='physical')
```

**How it works:**
```
PSF_size_μm = 54.3 μm (default, configurable)
xy_pixels = PSF_size_μm / dxy
z_pixels = PSF_size_μm / dz
```

**Results:**

| Resolution | PSF Pixels | Physical Size |
|------------|------------|---------------|
| 65nm/200nm | 837×837×273 | 54.4×54.4×54.6 μm ✓ |
| 130nm/300nm | 419×419×181 | 54.5×54.5×54.3 μm ✓ |
| 200nm/500nm | 273×273×109 | 54.6×54.6×54.5 μm ✓ |

**Advantages:**
- ✅ Physically consistent (cubic PSF in real space)
- ✅ Both lateral and axial scale with resolution
- ✅ More memory efficient with coarse pixels
- ✅ Captures same physical extent regardless of sampling

**Use when:**
- Starting new projects
- Want consistent behavior across resolutions
- Memory efficiency matters

---

### 2. C Heuristic Mode (C Code Compatible)

**Matches deconwolf C code behavior** exactly.

```python
xy, z = dwpy.auto_psf_size(dxy, dz, NA, wvl, ni, mode='c_heuristic')
```

**How it works:**
```
Lateral: xy = 181 (fixed, C code default)
Axial: z = (181 * 300 / dz_nm) * 2 + 3
```

**Results:**

| Resolution | PSF Pixels | Physical Size |
|------------|------------|---------------|
| 65nm/200nm | 181×181×273 | 11.8×11.8×54.6 μm |
| 130nm/300nm | 181×181×183 | 23.5×23.5×54.9 μm |
| 200nm/500nm | 181×181×111 | 36.2×36.2×55.5 μm |

**Note:** Physical size varies with lateral pixel size!

**Advantages:**
- ✅ Exact C code compatibility
- ✅ Matches existing PSF_dapi.tif sizing
- ✅ Conservative lateral size (always 181)

**Disadvantages:**
- ❌ Physically inconsistent (lateral size varies with dxy)
- ❌ Wastes memory with fine pixels (181 pixels at 65nm = 11.8μm)

**Use when:**
- Need exact C code compatibility
- Comparing with C implementation results
- Following existing workflows

---

### 3. Manual Mode

**Explicit control** over PSF dimensions.

```python
xy, z = dwpy.auto_psf_size(
    dxy, dz, NA, wvl, ni,
    mode='manual',
    xy_size=201,
    z_size=101
)
```

**Advantages:**
- ✅ Full control
- ✅ Can match specific requirements
- ✅ Useful for experiments/debugging

**Use when:**
- Testing specific PSF sizes
- Matching a particular reference PSF
- Special requirements

---

## Using in Config Files

```yaml
psf:
  model: "gibson-lanni"
  sizing_mode: "physical"  # or "c_heuristic", "manual"
  physical_extent: 54.3    # μm (for 'physical' mode)
  xy_size: null            # For 'manual' mode: specify explicit size
  z_size: null             # For 'manual' mode: specify explicit size
```

### Examples

**Physical mode (recommended):**
```yaml
# configs/my_experiment.yaml
psf:
  sizing_mode: "physical"
  physical_extent: 54.3  # Standard
  # OR
  physical_extent: 80.0  # Larger for thick samples
```

**C heuristic mode (compatibility):**
```yaml
# configs/dapi_c_compatible.yaml
psf:
  sizing_mode: "c_heuristic"
  # Lateral always 181, axial auto-calculated
```

**Manual mode:**
```yaml
# configs/custom_size.yaml
psf:
  sizing_mode: "manual"
  xy_size: 201  # Explicit
  z_size: 151   # Explicit
```

---

## Comparison Table

| Aspect | Physical | C Heuristic | Manual |
|--------|----------|-------------|--------|
| **Lateral scaling** | ✓ Scales with dxy | ✗ Fixed (181) | User specifies |
| **Axial scaling** | ✓ Scales with dz | ✓ Scales with dz | User specifies |
| **Physical consistency** | ✓ Cubic in space | ✗ Varies | Depends |
| **C compatibility** | ✗ Different sizes | ✓ Exact match | Depends |
| **Memory efficiency** | ✓ Optimal | ❌ Wastes with fine pixels | Depends |
| **Use case** | New projects | C compatibility | Special needs |

---

## Recommendations

### For New Work: Use `physical` Mode
```yaml
psf:
  sizing_mode: "physical"
  physical_extent: 54.3
```

**Benefits:**
- Physically consistent behavior
- Scales properly with resolution
- Memory efficient

### For C Code Comparison: Use `c_heuristic` Mode
```yaml
psf:
  sizing_mode: "c_heuristic"
```

**Benefits:**
- Exact match with C implementation
- Direct comparison possible

### For Special Requirements: Use `manual` Mode
```yaml
psf:
  sizing_mode: "manual"
  xy_size: 201
  z_size: 101
```

---

## Code Examples

```python
import dwpy

# Physical mode (recommended)
xy, z = dwpy.auto_psf_size(
    dxy=0.065, dz=0.2,
    NA=1.4, wvl=0.52, ni=1.515,
    mode='physical'
)
# Returns: (837, 273) - maintains 54.3μm cube

# C heuristic (C code compatibility)
xy, z = dwpy.auto_psf_size(
    dxy=0.13, dz=0.3,
    NA=1.45, wvl=0.461, ni=1.512,
    mode='c_heuristic'
)
# Returns: (181, 183) - matches C code

# Manual (explicit control)
xy, z = dwpy.auto_psf_size(
    dxy=0.1, dz=0.2,
    NA=1.4, wvl=0.52, ni=1.515,
    mode='manual',
    xy_size=201, z_size=151
)
# Returns: (201, 151) - as specified
```

---

## Why the Inconsistency in C Code?

The C code uses:
- **Lateral**: Fixed 181 pixels (historical default)
- **Axial**: Scaled by resolution (`181*300/dz_nm`)

**Assumption from C code comment:**
> "Assume that samples are imaged with about the same thickness
> regardless of resolution"

The `181 × 300nm = 54.3μm` represents typical biological sample depth. The C code scales axial pixels to maintain this depth but doesn't scale lateral (oversight or backward compatibility).

**Physical mode fixes this** by scaling both dimensions.

---

## Choosing Physical Extent

The default 54.3 μm works for most biological samples:

| Sample Type | Typical Thickness | Recommended Extent |
|-------------|-------------------|-------------------|
| Cultured cells | 10-20 μm | 54.3 μm (default) ✓ |
| Tissue sections | 50-100 μm | 80-120 μm |
| Embryos/organoids | 100-300 μm | 150-200 μm |
| Whole organisms | 500+ μm | Custom |

**In config:**
```yaml
psf:
  sizing_mode: "physical"
  physical_extent: 100.0  # For thick samples
```

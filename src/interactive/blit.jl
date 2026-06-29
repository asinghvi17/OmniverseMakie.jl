# CPU blit (M5): update the image! plot's data Observable from a host frame.
# GLMakie re-uploads the texture on the data change (spike §5 — idiomatic Makie path).
#
# Orientation: our ovrtx frame is Matrix{RGBA{N0f8}}[H,W] with row 1 = top (right-side-up).
# Makie's image! plots with first dimension = x (horizontal) and second dimension = y
# (vertical, y increases upward in the default Axis convention).
#
# Transform needed to display frame rows as vertical top-to-bottom:
#   - permutedims(frame): [W,H] — swaps dims so frame rows become y-axis (second dim)
#   - reverse(..., dims=2): flip second dim to match y-up — data[col, k] = frame[H-k+1, col]
# Result: data[col, H] (high y = TOP) = frame[1, col] (red); data[col, 1] (low y = BOTTOM) = frame[H, col] (blue).
#
# Verified in Step 1 REPL:
#   img[1][] = EndPoints (x range), img[2][] = EndPoints (y range), img[3][] = Matrix{RGBA{N0f8}}
#   Data Observable index = [3].  `image!(ax, frame)` — x=img[1], y=img[2], data=img[3].

# single source for the host-frame → Makie-image orientation; cpu_blit!, interactive_display, and resize_viewport! must all use it
_orient_for_display(frame) = reverse(permutedims(frame), dims = 2)

"""
    cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}}) -> Nothing

Update the GLMakie `image!` plot's data Observable from a host frame, triggering a
texture re-upload (CPU blit, M5 §5).

The host frame is `[H, W]` top-left origin (row 1 = top).  The transform
`reverse(permutedims(frame), dims=2)` maps frame rows to Makie's y-axis so the
image appears right-side-up in the GLMakie window.
"""
function cpu_blit!(image_plot, frame::AbstractMatrix{RGBA{N0f8}})
    # [3] = data Observable (x=img[1], y=img[2], data=img[3]; verified Step 1 REPL)
    image_plot[3][] = _orient_for_display(frame)
    return nothing
end

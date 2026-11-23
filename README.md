# Landscape

Landscape is an open, from-scratch implementation of the core machinery of gradient-based learning: loss functions, gradients, Hessians, and the diagnostics that reveal what an optimizer is actually doing.

## Context

Modern libraries hide the optimization layer entirely. You call `.fit()`, it converges, and you move on. That is the right way to build products, but it leaves the underlying objects — the loss surface, the gradient, the Hessian, the condition number, the convergence rate — invisible.

Landscape exists to make those objects visible. Every function is written by hand, small enough to read in one sitting, and accompanied by the reasoning behind it. The point is not to replace existing tools, but to understand what they are doing.

## Goals

- Build the loss → gradient → Hessian chain from first principles, in readable Python.
- Expose the diagnostics that distinguish the different ways a model can fail: optimization, approximation, specification, and generalization.
- Make every concept testable — each function should demonstrate the idea it implements.
- Stay small and legible. No autodiff engines, no neural network frameworks, no attempts to clone existing libraries.

## Open for contribution

This project is open for anyone who wants to learn by building. Contributions are welcome in any form:

- Implementing a new model, loss function, or diagnostic.
- Adding tests that demonstrate a concept.
- Writing examples that show a failure mode in action.
- Improving documentation or the reasoning behind an implementation.
- Fixing bugs, simplifying code, or questioning the design.

There is no fixed roadmap. The scope is intentionally bounded to the concepts that make gradient-based learning legible, and anything that serves that goal is in scope.

## Purpose

Landscape is primarily an experimentation space. It is meant for trying things, breaking them, and understanding why they broke. It is not a production library, and it is not trying to be one. If you learn something by building a piece of it, it has done its job.

## License

MIT.

---
title: Reference
---

# Glossary
<!-- Should remain alphabetical -->
- **Axis**: One of the independent directions along which array positions are arranged; for example, in a table or 2D array, the rows and columns are the two axes.
- **Axis length**: The number of positions along one axis.
- **Index**: A value used to select a position; this can refer to several different kinds of positions, so the following specific terms are preferred when the context isn't clear:
  - **Logical index**: A position as it appears to someone using an array: one coordinate for each direction, counted from zero. A logical index does not describe _where_ an element is stored in memory, but rather _which_ element is being selected.
  - **Storage position**: A position within the array's underlying data structure, counted forward in elements from the beginning of that storage.
- **Index space**: All of the logical positions that an array-like object contains. In a rectangular array, a coordinate is in the index space when every component is within its corresponding axis length.
- **Rank**: The number of independent directions along which an array is arranged. A list has rank one, a table has rank two, etc. Also called "dimensionality".
- **Shape**: The ordered tuple of axis lengths for an array.

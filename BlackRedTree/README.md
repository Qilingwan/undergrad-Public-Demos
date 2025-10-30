## Red-Black Tree File Indexer 

### 1. Overview

(a) This project implements a Red-Black Tree (RBTree) in C for storing and managing key–value pairs dynamically.
(b) The system is designed to read all text files within a local directory (`files/`), store each file name as a key, and its content as the value.
(c) The tree maintains balance automatically after each insertion, ensuring efficient search and insertion operations.
(d) Each stage of the program prints informative outputs to indicate progress and errors.

### 2. File Structure

(a) `rbtree.h` — header file defining the RBTree data structures and function prototypes.
(b) `rbtree.c` — implementation of Red-Black Tree creation, insertion, and traversal logic.
(c) `main.c` — driver program that reads files from the `files/` directory and indexes them into the RBTree.
(d) `files/` — input directory for test text files to be automatically read and stored.

### 3. Implementation / Methods

#### 3.1 Tree Structure Definition

(a) Each node (`RbNode`) stores a `key` (file name), a `value` (file content), and a color (`RED` or `BLACK`).
(b) The `RbTree` structure contains only a pointer to its root node.
(c) Each node keeps track of parent, left child, and right child pointers for balancing operations.

#### 3.2 Insertion and Balancing

(a) The `insertNode()` function inserts a new node while maintaining binary search order by key comparison.
(b) Duplicate keys trigger a value update rather than a new insertion, ensuring unique key mapping.
(c) The `fixInsert()` function restores Red-Black Tree properties after every insertion through recoloring and rotations (`rotateLeft`, `rotateRight`).
(d) The root node is always forced to be black at the end of rebalancing.

#### 3.3 File Loading and Directory Management

(a) The `main()` program creates a `files/` directory (if it does not exist).
(b) It opens and scans the directory using `opendir()` and `readdir()`, skipping system entries (`.`, `..`).
(c) For each valid file, its contents are read via `readFile()` and inserted into the tree.
(d) If no files are found, the program outputs a user-friendly warning message.

#### 3.4 Searching and Printing

(a) `searchNode()` performs a standard binary search by comparing keys recursively.
(b) `inorderPrint()` prints the tree contents in sorted order by key using in-order traversal.
(c) Each line of output follows the format:  `filename -> filecontent`

#### 3.5 Deletion (Partial Implementation)

(a) `deleteNode()` locates the node to delete by key and prints an informative message.
(b) Actual structural deletion and rebalancing are not implemented in this version.
(c) Example output when deleting a non-existent key: `Node not found for deletion: example.txt`

#### 3.6 Memory Management

(a) `freeNode()` and `freeTree()` are used to recursively free all allocated memory.
(b) Each node’s key and value strings are freed safely before deallocating the node structure.
(c) The tree pointer itself is released at the end of execution.

### 4. Output Demonstrations

#### 4.1 Example 1 — Files Loaded Successfully

**Directory structure:**

```
files/
 ├── a.txt
 ├── b.txt
 └── c.txt
```

**Console Output:**

```
Reading files from directory...
Loaded: a.txt
Loaded: b.txt
Loaded: c.txt

RBTree Index (3 files):
a.txt -> [content of a.txt]
b.txt -> [content of b.txt]
c.txt -> [content of c.txt]
```

#### 4.2 Example 2 — Empty Directory

**Console Output:**

```
Reading files from directory...
No files found in 'files' directory.
Please create some test files in the 'files' directory.

RBTree Index (0 files):
```

#### 4.3 Example 3 — File Read Error

**Console Output:**

```
Cannot open file: missing.txt
Failed to read: missing.txt
```

#### 4.4 Example 4 — Duplicate Key Insertion

**Console Output:**

```
Loaded: test.txt
Loaded: test.txt
```

(The second insertion replaces the previous value associated with `test.txt`.)

### 5. Build and Run Instructions

(a) Use the following commands in a terminal (Linux/macOS):

```
gcc main.c rbtree.c -o rbtree
./rbtree
```

(b) On Windows, compile using MinGW:

```
gcc main.c rbtree.c -o rbtree.exe
rbtree.exe
```

(c) Ensure that the `files/` directory exists before running the executable.

### 6. Notes and Future Work

(a) The deletion function currently only locates and reports the target node; full removal logic can be added using standard Red-Black deletion rules.
(b) A logging system or colored console output can be added for enhanced readability.
(c) The current version assumes UTF-8 text input; handling for binary files may be added in future iterations.
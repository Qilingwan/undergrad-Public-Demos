#ifndef RBTREE_H
#define RBTREE_H

#include <stdbool.h>

typedef enum { RED, BLACK } NodeColor;

typedef struct rbNode {
    char *key;               
    char *value;          
    NodeColor color;
    struct rbNode *left;
    struct rbNode *right;
    struct rbNode *parent;
} RbNode;

typedef struct {
    RbNode *root;
} RbTree;

RbTree *createTree();
void freeTree(RbTree *tree);
void insertNode(RbTree *tree, const char *key, const char *value);
bool deleteNode(RbTree *tree, const char *key);
char *searchNode(RbTree *tree, const char *key);
void inorderPrint(RbTree *tree);

#endif

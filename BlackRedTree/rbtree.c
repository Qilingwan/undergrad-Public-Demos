#include "rbtree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static RbNode *createNode(const char *key, const char *value, NodeColor color) {
    RbNode *node = malloc(sizeof(RbNode));
    node->key = strdup(key);
    node->value = strdup(value);
    node->color = color;
    node->left = node->right = node->parent = NULL;
    return node;
}

static void rotateLeft(RbTree *tree, RbNode *x) {
    RbNode *y = x->right;
    x->right = y->left;
    if (y->left) y->left->parent = x;
    y->parent = x->parent;
    if (!x->parent) tree->root = y;
    else if (x == x->parent->left) x->parent->left = y;
    else x->parent->right = y;
    y->left = x;
    x->parent = y;
}

static void rotateRight(RbTree *tree, RbNode *y) {
    RbNode *x = y->left;
    y->left = x->right;
    if (x->right) x->right->parent = y;
    x->parent = y->parent;
    if (!y->parent) tree->root = x;
    else if (y == y->parent->left) y->parent->left = x;
    else y->parent->right = x;
    x->right = y;
    y->parent = x;
}

static void fixInsert(RbTree *tree, RbNode *z) {
    while (z->parent && z->parent->color == RED) {
        if (z->parent == z->parent->parent->left) {
            RbNode *y = z->parent->parent->right;
            if (y && y->color == RED) {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    rotateLeft(tree, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                rotateRight(tree, z->parent->parent);
            }
        } else {
            RbNode *y = z->parent->parent->left;
            if (y && y->color == RED) {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rotateRight(tree, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                rotateLeft(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = BLACK;
}

RbTree *createTree() {
    RbTree *tree = malloc(sizeof(RbTree));
    tree->root = NULL;
    return tree;
}

void insertNode(RbTree *tree, const char *key, const char *value) {
    RbNode *z = createNode(key, value, RED);
    RbNode *y = NULL;
    RbNode *x = tree->root;
    while (x) {
        y = x;
        int cmp = strcmp(z->key, x->key);
        if (cmp < 0) x = x->left;
        else if (cmp > 0) x = x->right;
        else {
            char *new_value = strdup(value);
            if (new_value) {
                free(x->value);
                x->value = new_value;
            }

            free(z->key);
            free(z->value);
            free(z);
            return;
        }
    }
    z->parent = y;
    if (!y) tree->root = z;
    else if (strcmp(z->key, y->key) < 0) y->left = z;
    else y->right = z;
    fixInsert(tree, z);
}

static void inorderTraversal(RbNode *node) {
    if (!node) return;
    inorderTraversal(node->left);
    printf("%s -> %s\n", node->key, node->value);
    inorderTraversal(node->right);
}

void inorderPrint(RbTree *tree) {
    inorderTraversal(tree->root);
}

char *searchNode(RbTree *tree, const char *key) {
    RbNode *x = tree->root;
    while (x) {
        int cmp = strcmp(key, x->key);
        if (cmp == 0) return x->value;
        x = (cmp < 0) ? x->left : x->right;
    }
    return NULL;
}

bool deleteNode(RbTree *tree, const char *key) {
    RbNode *x = tree->root;
    while (x) {
        int cmp = strcmp(key, x->key);
        if (cmp == 0) {
            printf("Found node to delete: %s. (Full deletion not implemented)\n", key);
            return true;
        }
        x = (cmp < 0) ? x->left : x->right;
    }
    printf("Node not found for deletion: %s\n", key);
    return false;
}

void freeNode(RbNode *node) {
    if (!node) return;
    freeNode(node->left);
    freeNode(node->right);
    free(node->key);
    free(node->value);
    free(node);
}

void freeTree(RbTree *tree) {
    freeNode(tree->root);
    free(tree);
}

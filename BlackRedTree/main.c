#include "rbtree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

char *readFile(const char *fileName) {
    FILE *f = fopen(fileName, "r");
    if (!f) {
        printf("Cannot open file: %s\n", fileName);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    if (size == 0) {
        printf("File is empty: %s\n", fileName);
        fclose(f);
        return NULL;
    }

    char *buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

const char *getFileName(const char *path) {
    const char *filename = strrchr(path, '/');
    if (filename) return filename + 1;
    filename = strrchr(path, '\\');  
    if (filename) return filename + 1;
    return path;  
}

void createDirectory(const char *path) {
    #ifdef _WIN32
    _mkdir(path);
    #else
    mkdir(path, 0755);
    #endif
}

int main() {
    RbTree *tree = createTree();
    createDirectory("files");

    const char *filePaths[] = {
        "files/test1.txt", 
        "files/test2.txt", 
        "files/test3.txt"
    };
    
    for (int i = 0; i < 3; i++) {
        char *content = readFile(filePaths[i]);
        if (content) {
            const char *fileName = getFileName(filePaths[i]);
            insertNode(tree, fileName, content);
            free(content);
        } else {
            printf("File not found: %s\n", filePaths[i]);
        }
    }

    printf("\nRBTree Index:\n");
    inorderPrint(tree);
    freeTree(tree);
    
    return 0;
}
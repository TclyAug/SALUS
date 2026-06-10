#include "misc/util/abc_global.h"
#include "bdd/cudd/cuddInt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    DdNode **items;
    size_t len;
    size_t cap;
} NodeVec;

static char *read_truth_table_source(const char *source) {
    FILE *fp;
    long raw_len;
    char *raw;
    char *filtered;
    size_t i, j;

    if (source[0] != '@') {
        char *copy = (char *)malloc(strlen(source) + 1);
        if (copy == NULL) {
            fprintf(stderr, "out of memory while copying truth table\n");
            exit(1);
        }
        strcpy(copy, source);
        return copy;
    }

    fp = fopen(source + 1, "rb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open truth table file: %s\n", source + 1);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek truth table file\n");
        exit(1);
    }
    raw_len = ftell(fp);
    if (raw_len < 0) {
        fprintf(stderr, "cannot stat truth table file\n");
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot rewind truth table file\n");
        exit(1);
    }

    raw = (char *)malloc((size_t)raw_len + 1);
    filtered = (char *)malloc((size_t)raw_len + 1);
    if (raw == NULL || filtered == NULL) {
        fprintf(stderr, "out of memory while reading truth table file\n");
        exit(1);
    }
    if (fread(raw, 1, (size_t)raw_len, fp) != (size_t)raw_len) {
        fprintf(stderr, "cannot read truth table file\n");
        exit(1);
    }
    fclose(fp);
    raw[raw_len] = '\0';

    for (i = 0, j = 0; i < (size_t)raw_len; ++i) {
        if (raw[i] == '0' || raw[i] == '1') {
            filtered[j++] = raw[i];
        }
    }
    filtered[j] = '\0';
    free(raw);
    return filtered;
}

static void vec_push(NodeVec *vec, DdNode *node) {
    if (vec->len == vec->cap) {
        size_t new_cap = vec->cap == 0 ? 16 : vec->cap * 2;
        DdNode **new_items = (DdNode **)realloc(vec->items, new_cap * sizeof(DdNode *));
        if (new_items == NULL) {
            fprintf(stderr, "out of memory while growing node vector\n");
            exit(1);
        }
        vec->items = new_items;
        vec->cap = new_cap;
    }
    vec->items[vec->len++] = node;
}

static int vec_contains(const NodeVec *vec, DdNode *node) {
    size_t i;
    for (i = 0; i < vec->len; ++i) {
        if (vec->items[i] == node) {
            return 1;
        }
    }
    return 0;
}

static int is_constant_edge(DdNode *edge) {
    return Cudd_IsConstant(Cudd_Regular(edge));
}

static DdNode *build_from_truth_table(
    DdManager *dd,
    const char *truth_table,
    size_t table_len,
    int variable_index
) {
    size_t i;
    int all_zero = 1;
    int all_one = 1;

    for (i = 0; i < table_len; ++i) {
        if (truth_table[i] == '1') {
            all_zero = 0;
        } else if (truth_table[i] == '0') {
            all_one = 0;
        } else {
            fprintf(stderr, "truth table must contain only '0' or '1'\n");
            exit(1);
        }
    }

    if (all_zero) {
        return Cudd_ReadLogicZero(dd);
    }
    if (all_one) {
        return Cudd_ReadOne(dd);
    }
    if (variable_index < 0) {
        fprintf(stderr, "ran out of variables before truth table reduced to constant\n");
        exit(1);
    }

    {
        size_t split = table_len / 2;
        DdNode *low = build_from_truth_table(dd, truth_table, split, variable_index - 1);
        Cudd_Ref(low);
        DdNode *high = build_from_truth_table(dd, truth_table + split, split, variable_index - 1);
        Cudd_Ref(high);
        DdNode *var = Cudd_bddIthVar(dd, variable_index);
        DdNode *node = Cudd_bddIte(dd, var, high, low);
        Cudd_Ref(node);

        Cudd_RecursiveDeref(dd, high);
        Cudd_RecursiveDeref(dd, low);
        Cudd_Deref(node);
        return node;
    }
}

static void emit_reachable_nodes_stable(FILE *out, DdNode *node, NodeVec *visited) {
    DdNode *regular = Cudd_Regular(node);
    DdNode *low;
    DdNode *high;

    if (Cudd_IsConstant(regular) || vec_contains(visited, regular)) {
        return;
    }

    vec_push(visited, regular);
    low = cuddE(regular);
    high = cuddT(regular);

    emit_reachable_nodes_stable(out, low, visited);
    emit_reachable_nodes_stable(out, high, visited);

    fprintf(
        out,
        "node %llu %u %lld %d %lld %d\n",
        (unsigned long long)(uintptr_t)regular,
        regular->index,
        is_constant_edge(low) ? -1LL : (long long)(uintptr_t)Cudd_Regular(low),
        Cudd_IsComplement(low),
        is_constant_edge(high) ? -1LL : (long long)(uintptr_t)Cudd_Regular(high),
        Cudd_IsComplement(high)
    );
}

int main(int argc, char **argv) {
    int num_vars;
    char *truth_table;
    size_t expected_len;
    DdManager *dd;
    DdNode *root;
    NodeVec visited = {0};

    if (argc != 3) {
        fprintf(stderr, "usage: %s <num_vars> <truth_table_bits>\n", argv[0]);
        return 1;
    }

    num_vars = atoi(argv[1]);
    truth_table = read_truth_table_source(argv[2]);
    expected_len = ((size_t)1) << num_vars;

    if ((int)strlen(truth_table) != (int)expected_len) {
        fprintf(stderr, "truth table length must be exactly 2^num_vars\n");
        return 1;
    }

    dd = Cudd_Init(num_vars, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    Cudd_AutodynEnable(dd, CUDD_REORDER_SYMM_SIFT);

    root = build_from_truth_table(dd, truth_table, expected_len, num_vars - 1);
    Cudd_Ref(root);
    Cudd_ReduceHeap(dd, CUDD_REORDER_SYMM_SIFT, 1);

    printf("num_vars %d\n", num_vars);
    printf("truth_table %s\n", truth_table);
    printf(
        "root %lld %d\n",
        Cudd_IsConstant(Cudd_Regular(root)) ? -1LL : (long long)(uintptr_t)Cudd_Regular(root),
        Cudd_IsComplement(root)
    );
    emit_reachable_nodes_stable(stdout, root, &visited);

    free(truth_table);
    free(visited.items);
    Cudd_RecursiveDeref(dd, root);
    Cudd_Quit(dd);
    return 0;
}

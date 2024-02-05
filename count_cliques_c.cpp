#include <fstream>
#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <cstring>

using namespace std;

const int MAX = 10000;
unsigned long long global_count;
int graph[MAX][MAX];
int store[MAX];
int n, k, missing_edge_allowance;

// Check if vertices in store[0:b] are a clique
bool is_clique(int b) {
	int missing_edges = 0;
	for (int i = 0; i < b; i++) {
		for (int j = i + 1; j < b; j++) {
			if (graph[store[i]][store[j]] == 0) {
				missing_edges++;
			}
			if (graph[store[j]][store[i]] == 0) {
				missing_edges++;
			}
			if (missing_edges > missing_edge_allowance) {
				return false;
			}
		}
	}
	return true;
}

// Return the number of cliques containing vertices in store[0:l] by inserting vertices from [i, )
// i is the next available vertex to insert
// l is the current clique length
unsigned long long count_cliques(int i, int l) {
	if (l == k) {
		assert(is_clique(k));
		global_count += 1;
		cerr << global_count << endl;
		return 1;
	}
	unsigned long long count = 0;
	for (int j = i; j <= n - (k - l); j++) {
		store[l] = j;
		if (is_clique(l + 1)) {
			count += count_cliques(j+1, l+1);
		}
	}
	return count;
}

void read_graph(string fname) {
	ifstream graph_file;
	graph_file.open(fname);

	string line;
	getline(graph_file, line);
	n = stoi(line);
	getline(graph_file, line);
	k = stoi(line);
	getline(graph_file, line);
	double gamma = stod(line);
	missing_edge_allowance = int((1 - gamma) * (k * (k-1)));

	assert(n <= MAX);

	while (getline(graph_file, line)) {
		size_t idx1 = line.find(' ');
		int i = stoi(line.substr(0, idx1));
		int j = stoi(line.substr(idx1+1, line.length() - (idx1 + 1))); // idx1 + (len - idx1 - 1) + 1 = len
		assert((i >= 0) && (i < n) && (j >= 0) && (j < n)); // 0-indexed
		graph[i][j] = 1;
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			assert(graph[i][j] == 0 || graph[i][j] == 1);
		}
	}
}

int main(int argc, char* argv[]) {
	memset(graph, 0, (MAX * MAX * sizeof(int)));
	memset(store, 0, (MAX * sizeof(int)));

	assert(argc == 2);
	string graph_fname(argv[1]);
	read_graph(graph_fname);

	global_count = 0;
	unsigned long long count = count_cliques(0, 0);
	assert(count == global_count);
	cout << count << endl;
	return 0;
}

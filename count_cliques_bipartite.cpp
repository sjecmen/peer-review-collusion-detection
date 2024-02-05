#include <fstream>
#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <cstring>

using namespace std;

const int MAX = 10000;
int num_bids[MAX][MAX];
int num_possible[MAX][MAX];
const int HIST_BUCKETS = 11;
unsigned long long density_hist[HIST_BUCKETS]; // [0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [1, 1]
int store[MAX];
int n, k;
double g;

// Compute bid density of vertices in store[0:b] 
double bid_density(int b) {
	int total_bids = 0;
	int total_possible = 0;
	for (int i = 0; i < b; i++) {
		for (int j = i + 1; j < b; j++) {
			total_bids += num_bids[store[i]][store[j]] + num_bids[store[j]][store[i]];
			total_possible += num_possible[store[i]][store[j]] + num_possible[store[j]][store[i]];
		}
	}
	if (total_possible == 0) {
		return 0;
	}
	return total_bids / ((double) total_possible);

}

// Check if vertices in store[0:b] are a clique
bool is_clique(int b) {
	return (b == 1) or (bid_density(b) >= g); // length=1 is a clique
}

// Return the number of cliques containing vertices in store[0:l] by inserting vertices from [i, )
// i is the next available vertex to insert
// l is the current clique length
unsigned long long count_cliques(int i, int l) {
	if (l == k) {
		double d = bid_density(k);

		int bucket = floor(d * (HIST_BUCKETS - 1));
		//if (bucket == -1) {
		//	assert(s == 0);
		//	bucket = 0;
		//}
		assert(bucket >= 0 && bucket < HIST_BUCKETS);
		density_hist[bucket] += 1;

		for (int i = 0; i < HIST_BUCKETS; i++) {
			cerr << density_hist[i] << ' ';
		}
		cerr << endl;

		//bool r = is_clique(k);
		//assert(g < 1 || r);
		return 1;
	}
	unsigned long long count = 0;
	for (int j = i; j <= n - (k - l); j++) {
		store[l] = j;
		//if (g < 1 || is_clique(l + 1)) {
		count += count_cliques(j+1, l+1);
		//}
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
	g = stod(line);
	assert(g == 1);
	assert(n <= MAX);

	int num_lines = 0;
	while (getline(graph_file, line)) {
		num_lines += 1;
		size_t idx1 = line.find(' ');
		size_t idx2 = line.find(' ', idx1+1);
		size_t idx3 = line.find(' ', idx2+1);
		int i = stoi(line.substr(0, idx1));
		int j = stoi(line.substr(idx1+1, idx2 - (idx1+1)));
		int b = stoi(line.substr(idx2+1, idx3 - (idx2+1)));
		int p = stoi(line.substr(idx3+1, line.length() - (idx3+1))); // idx1 + (idx2 - idx1-1) + (idx3 - idx2-1) + (len -idx3-1) + 3 = len
		assert((i >= 0) && (i < n) && (j >= 0) && (j < n)); // 0-indexed
		assert(b <= p);
		num_bids[i][j] = b;
		num_possible[i][j] = p;
	}
	assert(num_lines == n*n);
}


int main(int argc, char* argv[]) {
	memset(num_bids, 0, (MAX * MAX * sizeof(int)));
	memset(num_possible, 0, (MAX * MAX * sizeof(int)));
	memset(density_hist, 0, (HIST_BUCKETS * sizeof(unsigned long long)));
	memset(store, 0, (MAX * sizeof(int)));

	assert(argc == 2);
	string graph_fname(argv[1]);
	read_graph(graph_fname);

	unsigned long long count = count_cliques(0, 0);
	cout << count << endl;

	unsigned long long text_count = 0;
	for (int i = 0; i < HIST_BUCKETS; i++) {
		cout << density_hist[i] << endl;
		text_count += density_hist[i];
	}
	assert(count == text_count);
	return 0;
}

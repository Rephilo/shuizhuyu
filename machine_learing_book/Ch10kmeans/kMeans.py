from numpy import *


def load_data(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        curr_line = line.strip().split('\t')
        flt_line = map(float, curr_line)
        data_mat.append(flt_line)
    return data_mat


def distEclud(veca, vecb):
    return sqrt(sum(power(veca - vecb, 2)))


def rand_cent(data_set, k):
    n = shape(data_set)[1]
    cent_troids = mat(zeros((k, n)))
    for i in range(n):
        min_j = min(data_set[:j])
        range_j = float(max(data_set[:, j]) - min_j)
        cent_troids[:, j] = min_j + range_j * random.rand(k, 1)
    return cent_troids


def kMeans(data_set, k, dist_meas=distEclud(), create_cent=rand_cent):
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m, 2)))
    cent_troid = create_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(cent_troid[j, :], data_set[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        print(cent_troid)
        for cent in range(k):
            pts_in_clust = data_set[nonzero(cluster_assment[:, 0].A == cent)[0]]
            cent_troid[cent, :] = mean(pts_in_clust, axis=0)
    return cent_troid, cluster_assment


def bi_kmeans(data_set, k, dist_meas=distEclud):
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m, 2)))
    centroid0 = mean(data_set, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(mat(centroid0), data_set[j, :]) ** 2
    while len(cent_list) < k:
        lowest_sse = inf
        for i in range(len(cent_list)):
            pts_in_curr_cluster = data_set[nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_clust_ass = kMeans(pts_in_curr_cluster, 2, dist_meas)
            sse_split = sum(split_clust_ass[:, 1])
            sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print("sse_split, and not_split", sse_split, sse_not_split)

            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print('the best cent to split is: ', best_cent_to_split)
        print('the len of best clust ass is: ', len(best_clust_ass))
        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
        return mat(cent_list), cluster_assment

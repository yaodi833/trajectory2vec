from sklearn.cluster import *
import cPickle
f1 = file('./cluster_data/sim_data_out_test_normal_reverse')
out1 = cPickle.load(f1)
data = []
# for item in out1:
#     print item
for i in range(2000):
    data.append(out1[i][0][0])
print len(data)
num_clusters = 4
# km = KMeans(n_clusters=num_clusters,random_state=2016)
# km = MeanShift()
# km = Birch(n_clusters=num_clusters)
km = AgglomerativeClustering(n_clusters=num_clusters)
# km = MiniBatchKMeans(n_clusters=num_clusters,random_state=2016,batch_size=100)

# km = DBSCAN(eps=5, min_samples=4)
km.fit(data)
clusters = km.labels_.tolist()
print clusters
print clusters.count(0)
print clusters.count(1)
print clusters.count(2)
print clusters.count(3)
count_times = []
print f1
all = 0.
m = 0.
item = set(clusters[:500])
l = []
for i in item:
    l.append(clusters[:500].count(i))
print 'Straight'
m = max(l)
all = all + m
print float(m)/500


m = 0.
item = set(clusters[500:1000])
l = []
for i in item:
    l.append(clusters[500:1000].count(i))
print 'Circling'
m = max(l)
all = all + m
print float(m)/500

m = 0.
item = set(clusters[1000:1500])
l = []
for i in item:
    l.append(clusters[1000:1500].count(i))
m = max(l)
print 'motionless'
all = all + m
print float(m)/500


m = 0.
item = set(clusters[1500:2000])
l = []
for i in item:
    l.append(clusters[1500:2000].count(i))
m = max(l)
print 'bending'
all = all + m
print float(m)/500
print 'overall'
print all/2000


# for i in range(50,100):
#     pass
# for i in range()

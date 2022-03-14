import tensorflow as tf

class manyVariables:
    def __init__(self):
        self.initList = [None]*100
        for i in range(100):
            self.initList[i] = tf.Variable(tf.random.normal([5,5]))
        self.makeSomeMoreVariables()
        self.ckpt = self.makeCheckpoint()

    def makeSomeMoreVariables(self):
        self.moreList = [None]*10
        for i in range(10):
            self.moreList[i] = tf.Variable(tf.random.normal([3,3]))

    def makeCheckpoint(self):
        return tf.train.Checkpoint(
            init3=self.initList[3], init55=self.initList[55],
            init60=self.initList[60], more4=self.moreList[4])

    def saveVariables(self):
        self.ckpt.save('./ckpt')

    def restoreVariables(self):
        status = self.ckpt.restore(tf.train.latest_checkpoint('.'))
        status.assert_consumed()  # Optional check

# Create variables
v1 = manyVariables()
# Assigned fixed values
for i, v in enumerate(v1.initList):
    v.assign(i * tf.ones_like(v))
for i, v in enumerate(v1.moreList):
    v.assign(100 + i * tf.ones_like(v))
# Save them
v1.saveVariables()

# Create new variables
v2 = manyVariables()
# Check initial values
print(v2.initList[2].numpy())
# [[-1.9110833   0.05956204 -1.1753829  -0.3572553  -0.95049495]
#  [ 0.31409055  1.1262076   0.47890127 -0.1699607   0.4409122 ]
#  [-0.75385517 -0.13847834  0.97012395  0.42515194 -1.4371008 ]
#  [ 0.44205236  0.86158335  0.6919655  -2.5156968   0.16496429]
#  [-1.241602   -0.15177743  0.5603795  -0.3560254  -0.18536267]]
print(v2.initList[3].numpy())
# [[-3.3441594  -0.18425298 -0.4898144  -1.2330629   0.08798431]
#  [ 1.5002227   0.99475247  0.7817361   0.3849587  -0.59548247]
#  [-0.57121766 -1.277224    0.6957546  -0.67618763  0.0510064 ]
#  [ 0.85491985  0.13310803 -0.93152267  0.10205163  0.57520276]
#  [-1.0606447  -0.16966362 -1.0448577   0.56799036 -0.90726566]]

# Restore them
v2.restoreVariables()
# Check values after restoring
print(v2.initList[2].numpy())
# [[-1.9110833   0.05956204 -1.1753829  -0.3572553  -0.95049495]
#  [ 0.31409055  1.1262076   0.47890127 -0.1699607   0.4409122 ]
#  [-0.75385517 -0.13847834  0.97012395  0.42515194 -1.4371008 ]
#  [ 0.44205236  0.86158335  0.6919655  -2.5156968   0.16496429]
#  [-1.241602   -0.15177743  0.5603795  -0.3560254  -0.18536267]]
print(v2.initList[3].numpy())
# [[3. 3. 3. 3. 3.]
#  [3. 3. 3. 3. 3.]
#  [3. 3. 3. 3. 3.]
#  [3. 3. 3. 3. 3.]
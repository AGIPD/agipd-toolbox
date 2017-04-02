analogData = analog[:,0,1,5]
darkOffset = getDarkCalInSitu(analogData)

plt.plot(analogData, '.')
plt.axhline(darkOffset)
plt.show()
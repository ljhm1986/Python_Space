import math
class Stats:
    @staticmethod
    def sum(*arg):
        hap = 0
        for i in arg:
            hap += i
        return hap
    
    @staticmethod
    def mean(*arg):
        return Stats.sum(*arg) / len(arg)
    @staticmethod
    def variance(*arg):
        total = 0
        m = Stats.mean(*arg)
        for i in arg:
            total += (i-m)**2
        return total/(len(arg)-1)
    
    @staticmethod
    def stddev(*arg):
        return math.sqrt(Stats.variance(*arg))

#파일을 불러들인 다음에 파일을 테스트할 방법 보여줌 
#라이브러리로 import 해서 불러들일때에는 실행되지 않음 
if __name__=="__main__":
    from Stats_class import Stats
    print(Stats.sum(1,2,3,4,5))
    print(Stats.mean(1,2,3,4,5))
    print(Stats.variance(1,2,3,4,5))
    print(Stats.stddev(1,2,3,4,5))
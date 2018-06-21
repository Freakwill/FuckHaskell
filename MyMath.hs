
{-
MyMath: maths functions
-}

module MyMath
( 
   sgn,
   hardlim,
   hardlims,
   sigmoid,
   dsigmoid,
   dtanh,
   spiky,
   gauss,
   gauss_std,
   same_sign,
   trapint
) where 

-- type OneDimFunc a = Num a => a -> a

type RealOneDimFunc = Double -> Double

sgn :: RealOneDimFunc
sgn x | x<0 = -1
      | x>0 = 1
      | x==0 = 0


hardlim :: RealOneDimFunc
hardlim x | x < 0 = 0
          | x == 0 = 0
          | x > 0 = 1


hardlims :: RealOneDimFunc
hardlims x | x < 0 = -1
          | x == 0 = 0
          | x > 0 = 1

sigmoid :: RealOneDimFunc
sigmoid x = 1 / (1 + exp (-x))

dsigmoid :: RealOneDimFunc
dsigmoid x = y * (1 - y)
    where y = sigmoid x


dtanh :: RealOneDimFunc
dtanh x = (1+y) * (1-y)
    where y = tanh x

spiky :: Double -> Double -> Double
spiky x p | x == 0 =0
        | x /= 0 =abs(x)**(-1 / (2*p))

gauss :: Double -> Double -> RealOneDimFunc
gauss mu sigma = \x -> exp(-(x - mu)**2 / (2 * sigma**2))

gauss_std :: RealOneDimFunc
gauss_std x = gauss x 0 1

same_sign :: Double -> Double -> Bool
same_sign a b = (a<0 && b<0) || (a>0 && b>0) || (a==0 || b==0)



dilation :: Double -> RealOneDimFunc -> RealOneDimFunc
dilation x f y = x * (f x*y)


trapint :: RealOneDimFunc -> Double -> Double -> Int -> Double
trapint f a b n = if n == 0 then ((f a) + (f b)) * (b - a) / 2
               else let m = (a + b) / 2 in (trapint f a m n-1) + (trapint f m b n-1)

antider :: RealOneDimFunc -> RealOneDimFunc
antider f = \x -> trapint f 0 x 10

err :: RealOneDimFunc
err = (*(2/pi)) . (antider gauss_std)


-- fejer :: Int -> RealOneDimFunc
-- fejer n t
--     | n == 0 = 1
--     | let tol=0.000001 in abs (floor t - t) < tol = n + 1
--     | otherwise = (sin ((n + 1) * pi * t)) / ((sin (pi * t))^2 * (n + 1))

-- bump :: Double -> Double -> RealOneDimFunc
--     -- bump type function
--     -- int(bump(lb,ub)) == (ub-lb)/2 * I, I=0.444
--     -- == (2*x-lb-ub)/(ub-lb)
-- bump lb ub x 
--     | (x <= -1 || x >= 1) = 0
--     | otherwise = exp (-1/(1 - x^2))
--     where x = 2*(x-lb) / (ub-lb)-1


main :: IO ()
main = do 
   putStrLn "The factorial of 5 is:"  
   print (bump 0 2 1)


{-


def dbump(x, lb=-1, ub=1):
    # bump type function
    # int(bump(lb,ub)) == (ub-lb)/2 * I, I=0.444
    x = 2*(x-lb)/(ub-lb)-1
    if x <= -1 or x >= 1 or x == 0:
        return 0
    else:
        return -np.exp(-1/(1-x**2)) * 2*x / (1-x**2)**2 / (ub-lb)


def dirichlet(t, n=1):
    if n == 0:
        return 1
    if np.floor(t) == t:
        return 2 * n + 1
    else:
        return np.sin((2 * n + 1) * np.pi * t) / np.sin(np.pi * t)


def dirichletez(t, n=1):
    if n == 0:
        return 1
    if np.floor(t) == t:
        return 2 * n + 1
    else:
        return np.sin((2 * n + 1) * np.pi * t) / (t * np.pi)

-}

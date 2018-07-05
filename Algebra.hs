{-# LANGUAGE InstanceSigs #-}

module Algebra
( 
   getTail
) where

import Numeric.Algebra.Class
import Numeric.Algebra

newtype Polynomial = Polynomial {getCoef :: [Integer]} deriving (Eq, Show)

zero = Polynomial []

get :: Polynomial -> Int -> Integer
get p n = (getCoef p) !! n
getTail :: Polynomial -> [Integer]
getTail p = tail (getCoef p)
split :: Polynomial -> (Integer, [Integer])
split p = let (a,r)=Prelude.splitAt 1 (getCoef p) in (a !! 0, r)

-- leading :: Polynomial -> Polynomial
-- leading zero = zero
-- leading (Polynomial as) = Polynomial (head as : (replicate ((length as) Prelude.-1) 0))

-- degree :: Polynomial -> Int
-- degree zero = -1
-- degree p = (length (dropWhile (== 0) (getCoef p))) Prelude.- 1

sm :: Integer -> String -> String
sm 1 s = s
sm (-1) s = "-" ++ s
sm a s = (Prelude.show a) ++ s

show (Polynomial []) = Prelude.show 0
show (Polynomial [x]) = Prelude.show x
show (Polynomial [x, 0]) = if x==0 then (Prelude.show 0) else (sm x "x + ")
show (Polynomial [x, y]) = if x==0 then (Prelude.show y) else (sm x "x + ") ++ (Prelude.show y)
show p@(Polynomial (a:r)) = (sm a "x^") ++ (Prelude.show (length r)) ++ (Algebra.show1 (Polynomial r))
show1 p@(Polynomial (a:r))
    |a Prelude.>0 = " + " ++ (Algebra.show p)
    |a Prelude.==0 = " + " ++ (Algebra.show (Polynomial r))
    | otherwise = " - " ++ (Algebra.show (Polynomial (abs a : r)))

-- solve :: Polynomial -> Maybe (Double, Double)
-- solve (Polynomial [a, b, c]) = let delta=(b Prelude.^2 Prelude.- 4 Prelude.* a Prelude.* c) in
--     if delta Prelude.<0 
--         then Nothing 
--     else 
--         Just ((b Prelude.- (sqrt delta)) Prelude./ (2 Prelude.* a), (b Prelude.+ (sqrt delta)) Prelude./ (2 Prelude.* a))


-- instance Semiring Polynomial where

--     _ + zero = _
--     zero + _ = _

--     p + q = let d=(((Prelude.-) `on` degree) p q) in if (d Prelude.< 0) then ((Prelude.+) <$> ((replicate d 0) ++ p) q) else q Prelude.+ p

--     _ * zero = zero
--     zero * _ = zero

--     p * q = let (a, r)=(Prelude.splitAt p) in ((Prelude.* a) <$> (getCoef q)) ++ (replicated (degree p) 0) Prelude.+ (Polynomial r) Prelude.* q
    
    
main = do 
   putStrLn "The factorial of 5 is:"  
   print (Algebra.show (Polynomial [5, -1, 4]))



{-
MyMath: maths functions
-}

module MyRandom
( 
   
) where 

-- type OneDimFunc a = Num a => a -> a

import Control.Monad
import Control.Monad.Random

main :: IO ()

main = do
  rs <- replicateM 10 (getRandom :: IO Int)  -- [IO Int] -> IO [Int]
  print rs

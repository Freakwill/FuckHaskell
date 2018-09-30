module MyMonadTrans
( 

)
where

import Control.Monad.Trans.Class
import Control.Monad.Trans.State.Strict
import Control.Monad.Random
import Data.List
import System.Random


calculator :: StateT Double IO ()
calculator = do
  result <- get
  lift $ print result
  (op:input) <- lift getLine
  let opFn = case op of
          '+' -> sAdd
          '-' -> sMinus
          _ -> keep in
    opFn $ read input
  where
    sAdd x = modify (+ x)
    sMinus x = modify (\y -> y - x)
    keep = const $ lift $ print "input +/ number"

main :: IO ((), Double)
computer = forever calculator :: StateT Double IO ()
main = runStateT computer 0
module MyMonad

where

import Control.Monad.State.Lazy

type Calculator = State Double (Double -> Double)  -- State ouput state

(~+) :: Double -> Calculator
(~+) x = state $ \s -> ((+x), s + x)

(~~) :: (Double -> Double) -> Calculator
(~~) f = state $ \s -> (f, f s)

(~-) :: Double -> Calculator
(~-) x = (~~) (\s -> s - x)


op :: State Double (Double -> Double)
op = do
    (~+) 10
    (~-) 1
    (~~) (*2)
    >>= (~~)
    >>= (~~)

main :: IO ()
main = do 
    let (_, res) = runState op 0 in putStrLn ("WoW " ++ (show res))


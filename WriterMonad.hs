-- My example, an  elegant calculator

module StateMonad
( 
   Log,  
)
where

import qualified Data.Strings as S
import Control.Monad.Writer.Lazy

type Log a = Writer String a

acc :: String -> Log String
acc a = writer $ (a, a ++ "Hello")

greeting :: String -> Log String
greeting name = writer $ ("Hello " ++ name, "--say hello to " ++ name)

fucking :: String -> Log String
fucking name = writer $ ("Fuck " ++ name, "--just fuck " ++ name)

hline = tell "\r\n"

end :: String -> String
end s= s ++ "[THE END]"

op :: Log String
op = do 
    a <- writer $ ("Lucy", "1")
    a <- greeting a
    hline
    a <- (writer $ ("Lily", "2")) >>= greeting
    hline
    fucking "bitch"
    return . end $ a

main :: IO ()
main = do
    let (_, res) = runWriter op in putStrLn res

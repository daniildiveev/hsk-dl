module Error
  ( TensorError(..)
  , nyi
  ) where

data TensorError
  = ShapeMismatch String
  | OutOfBounds String
  | NonContiguousRequired String
  | BroadcastError String
  | NotImplemented String
  deriving (Eq, Show)

nyi :: String -> a
nyi what = error ("NYI: " <> what)

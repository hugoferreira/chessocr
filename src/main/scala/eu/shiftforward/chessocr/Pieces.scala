package eu.shiftforward.chessocr

object Pieces {
  val matrix =
    (1.0 :: 0.0 :: 0.0 :: 0.0 :: 0.0 :: 0.0 :: Nil) ::
    (0.0 :: 1.0 :: 0.0 :: 0.0 :: 0.0 :: 0.0 :: Nil) ::
    (0.0 :: 0.0 :: 1.0 :: 0.0 :: 0.0 :: 0.0 :: Nil) ::
    (0.0 :: 0.0 :: 0.0 :: 1.0 :: 0.0 :: 0.0 :: Nil) ::
    (0.0 :: 0.0 :: 0.0 :: 0.0 :: 1.0 :: 0.0 :: Nil) ::
    (0.0 :: 0.0 :: 0.0 :: 0.0 :: 0.0 :: 1.0 :: Nil) :: Nil

  val pieces = (matrix.map(1.0 :: _) ++ matrix.map(0.0 :: _)).map(_.toArray[Double])

  val (whitePawn, whiteTower, whiteBishop, whiteKnight, whiteQueen, whiteKing, blackPawn, blackTower, blackBishop, blackKnight, blackQueen, blackKing) =
      (pieces(0), pieces(1), pieces(2), pieces(3), pieces(4), pieces(5), pieces(6), pieces(7), pieces(8), pieces(9), pieces(10), pieces(11))
}